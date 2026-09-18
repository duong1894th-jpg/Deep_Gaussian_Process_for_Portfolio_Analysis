import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import warnings
from datetime import datetime
from joblib import Parallel, delayed

import torch
import gpytorch

warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURATION & PORTFOLIO
# ==========================================
PORTFOLIO_INPUT = [
    ('VCB.VN', 'BNK'),
    ('BID.VN', 'BNK'),
    ('CTG.VN', 'BNK'),
    ('TCB.VN', 'BNK'),
    ('MBB.VN', 'BNK'),
    ('VPB.VN', 'BNK'),
    ('ACB.VN', 'BNK'),
    ('STB.VN', 'BNK'),
    ('HDB.VN', 'BNK'),
    ('VIB.VN', 'BNK'),
    ('SHB.VN', 'BNK'),
    ('TPB.VN', 'BNK'),
]

TYPE = 'Open'
START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
RANDOM_SEED = 42

# Strategy Parameters
HORIZON_BLOCKS = 10
N_ITER_BO = 50 # DeepGP epochs per chunk

# External Feature Mapping
EXT_MAP = {
    'BNK': ['USDVND=X', '^TNX', '^IRX'],
    'STL': ['HRC=F', 'CL=F', 'BDRY'],
    'DLY': ['ZC=F', 'ZM=F', 'CL=F'],
    'SFD': ['USDVND=X', 'BDRY', 'CL=F'],
    'SMC': ['^IXIC', 'SOXX', 'HG=F', '^TWII', 'DX-Y.NYB'],
    'REA': ['USDVND=X', 'HRC=F', '^TNX'],
    'ENG': ['BZ=F', 'NG=F', 'DX-Y.NYB'],
    'FTL': ['NG=F', 'ZR=F', 'ZC=F', 'ZW=F'],
    'RTL': ['USDVND=X', 'CL=F', 'DX-Y.NYB', 'BDRY', 'TIP', 'XLY', 'VNM'],
    'SHP': ['BDRY', 'CL=F'],
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================
# 1. Deep GP Core Engine (DKL via GPyTorch)
# ==========================================
class ContextFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, extract_dim=4):
        super(ContextFeatureExtractor, self).__init__()
        # The NN processes both internal and external (context) features together
        self.add_module('linear1', torch.nn.Linear(data_dim, 16))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(16, extract_dim))

class DeepKernelModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(DeepKernelModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        # Apply RBF Kernel on the Neural Network's extracted features
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_dkl(train_x, train_y, epochs=N_ITER_BO):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    feature_extractor = ContextFeatureExtractor(train_x.size(-1))
    model = DeepKernelModel(train_x, train_y, likelihood, feature_extractor)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.05)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood

# ==========================================
# 2. Pipeline
# ==========================================
def process_single_stock(ticker, sector_codes):
    print(f"\nProcessing {ticker} (Sectors: {sector_codes})...")
    
    # Data Fetching
    stock_data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if stock_data.empty: return None
    if isinstance(stock_data.columns, pd.MultiIndex): stock_data.columns = stock_data.columns.get_level_values(0)
    
    # External features
    codes = [s.strip() for s in sector_codes.split(',') if s.strip()]
    ext_tickers = []
    for c in codes: ext_tickers.extend(EXT_MAP.get(c, []))
    ext_tickers = sorted(list(set(ext_tickers)))
    
    ext_dfs = {}
    if ext_tickers:
        raw_ext = yf.download(ext_tickers, start=START_DATE, end=END_DATE, progress=False)
        for t in ext_tickers:
            data = raw_ext[TYPE][t] if len(ext_tickers) > 1 else raw_ext[TYPE]
            if not data.dropna().empty: ext_dfs[t] = data.rename(t)

    def prepare_data(N):
        N = int(N)
        df_d = stock_data[[TYPE, "High", "Low", "Volume"]].copy()
        df_d['idx_p'] = df_d[TYPE]
        for t, data in ext_dfs.items(): df_d = df_d.join(data, how='left')
        
        df_d = df_d.ffill().dropna()
        if len(df_d) < N * 3: return pd.DataFrame(), [], []

        df_d["pct_r"] = df_d[TYPE].pct_change()
        ma20_p = df_d[TYPE].rolling(20).mean()
        ma20_v = df_d["Volume"].rolling(20).mean()
        tr = pd.concat([df_d["High"]-df_d["Low"], abs(df_d["High"]-df_d[TYPE].shift(1)), abs(df_d["Low"]-df_d[TYPE].shift(1))], axis=1).max(axis=1)
        df_d["vol_daily"] = df_d["pct_r"].rolling(60).std()
        
        df_d["r3"] = np.log(df_d[TYPE] / (df_d[TYPE].shift(3) + 1e-9))
        df_d["dt"] = (df_d[TYPE] - ma20_p) / (ma20_p + 1e-9)
        df_d["sigma_t"] = df_d["pct_r"].rolling(N).std()
        df_d["delta_sigma"] = df_d["sigma_t"] - df_d["sigma_t"].shift(5)
        df_d["natr"] = tr.rolling(N).mean() / (df_d[TYPE] + 1e-9)
        df_d["rv"] = df_d["Volume"] / (ma20_v + 1e-9)
        
        df_d = df_d.dropna()
        starts = np.arange(len(df_d) - N, -1, -N)[::-1]
        
        chunk_rows = []
        for i in range(1, len(starts)):
            block_prev = df_d.iloc[starts[i-1] : starts[i]]
            block_curr = df_d.iloc[starts[i] : starts[i] + N]
            c_ret = (block_prev[TYPE].iloc[-1] - block_prev[TYPE].iloc[0]) / (block_prev[TYPE].iloc[0] + 1e-9)
            
            row = {
                "Date": df_d.index[starts[i]], 
                "y": (block_curr[TYPE].iloc[-1] - block_curr[TYPE].iloc[0]) / (block_curr[TYPE].iloc[0] * block_curr["vol_daily"].iloc[0] * np.sqrt(N) + 1e-9),
                "f_r3": block_prev["r3"].iloc[-1], "f_dt": block_prev["dt"].iloc[-1], 
                "f_sigma": block_prev["sigma_t"].iloc[-1], "f_dsigma": block_prev["delta_sigma"].iloc[-1],
                "f_natr": block_prev["natr"].iloc[-1], "f_rv": block_prev["rv"].mean(), 
                "f_intensity": block_prev["rv"].mean() * abs(c_ret),
                "f_skew": np.nan_to_num(skew(block_prev["pct_r"]), nan=0.0), 
                "f_entropy": np.nan_to_num(-((block_prev["pct_r"]>0).mean()*np.log(np.clip((block_prev["pct_r"]>0).mean(),0.001,0.999)) + (1-(block_prev["pct_r"]>0).mean())*np.log(np.clip(1-(block_prev["pct_r"]>0).mean(),0.001,0.999))), nan=0.0)
            }
            for t in ext_tickers:
                if t in block_prev.columns:
                    row[f"ext_{t}"] = (block_prev[t].iloc[-1] - block_prev[t].iloc[0]) / (block_prev[t].iloc[0] + 1e-9)
            chunk_rows.append(row)
        
        df_chunks = pd.DataFrame(chunk_rows).replace([np.inf, -np.inf], np.nan).dropna()
        int_f = ["f_r3", "f_dt", "f_sigma", "f_dsigma", "f_natr", "f_rv", "f_intensity", "f_skew", "f_entropy"]
        ext_f = [f"ext_{t}" for t in ext_tickers if f"ext_{t}" in df_chunks.columns]
        return df_chunks, int_f, ext_f

    def evaluate_config(N):
        df, int_f, ext_f = prepare_data(N)
        if len(df) < HORIZON_BLOCKS + 5: return -1.0, -1.0, None
        
        horizon_indices = np.arange(len(df) - HORIZON_BLOCKS, len(df))
        mu_preds, actuals = [], []
        std_s = [0.0]
        
        for idx in horizon_indices:
            train_df, test_df = df.iloc[:idx], df.iloc[[idx]]
            features = int_f + ext_f
            
            sC = StandardScaler()
            Cti = sC.fit_transform(train_df[features]) if features else np.empty((len(train_df), 0))
            Yt = train_df["y"].values
            
            train_x = torch.tensor(Cti, dtype=torch.float32)
            train_y = torch.tensor(Yt, dtype=torch.float32)
            
            if len(train_x) < 10: 
                continue

            model, likelihood = train_dkl(train_x, train_y)

            model.eval()
            likelihood.eval()

            Csi = sC.transform(test_df[features]) if features else np.empty((1, 0))
            test_x = torch.tensor(Csi, dtype=torch.float32)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))
                mu_preds.append(observed_pred.mean.item())
                std_s = observed_pred.stddev.numpy()
                actuals.append(test_df["y"].values[0])
            
        if not mu_preds: return -1.0, -1.0, None

        hr = np.mean(np.sign(mu_preds) == np.sign(actuals)) * 100
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(mu_preds))**2))
        return hr, rmse, {"mu": mu_preds, "actual": actuals, "std": std_s[0]}

    # Optimization Logic (DeepGP learns the blending 'w', so we only tune N)
    N_range = [5, 10, 20, 30, 40, 60]
    
    best_score, best_N, best_stats = -1.0, None, None
    for n in N_range:
        hr, rmse, report = evaluate_config(n)
        if hr == -1.0: continue
        score = hr / (rmse + 1e-9) if hr >= 60 else -1.0
        if score > best_score:
            best_score, best_N, best_stats = score, n, (hr, rmse, report)
    
    if best_N is None:
        best_hr = -1.0
        for n in N_range:
            hr, rmse, report = evaluate_config(n)
            if hr > best_hr:
                best_hr, best_N, best_stats = hr, n, (hr, rmse, report)
        best_score = best_hr / (best_stats[1] + 1e-9)

    if best_N is None: return None

    # Future Forecast with Champion Deep GP
    final_df, int_f, ext_f = prepare_data(best_N)
    df_d = stock_data[[TYPE, "High", "Low", "Volume"]].copy()
    df_d["pct_r"] = df_d[TYPE].pct_change()
    latest_block = df_d.tail(int(best_N))
    ma20_p, ma20_v = df_d[TYPE].rolling(20).mean().iloc[-1], df_d["Volume"].rolling(20).mean().iloc[-1]
    tr = pd.concat([latest_block["High"]-latest_block["Low"], abs(latest_block["High"]-latest_block[TYPE].shift(1))], axis=1).max(axis=1)
    
    c_ret = (latest_block[TYPE].iloc[-1] - latest_block[TYPE].iloc[0]) / (latest_block[TYPE].iloc[0] + 1e-9)
    f_r3 = np.log(latest_block[TYPE].iloc[-1] / (latest_block[TYPE].iloc[-4] + 1e-9))
    f_dt = (latest_block[TYPE].iloc[-1] - ma20_p) / (ma20_p + 1e-9)
    f_sigma = latest_block["pct_r"].std()
    f_dsigma = f_sigma - df_d["pct_r"].shift(5).tail(int(best_N)).std()
    f_natr = tr.mean() / (latest_block[TYPE].iloc[-1] + 1e-9)
    f_rv = latest_block["Volume"].mean() / (df_d["Volume"].mean() + 1e-9)
    f_intensity = f_rv * abs(c_ret)
    f_skew = np.nan_to_num(skew(latest_block["pct_r"].dropna()), nan=0.0)
    p_up = np.clip((latest_block["pct_r"] > 0).mean(), 0.001, 0.999)
    f_entropy = -p_up * np.log(p_up) - (1-p_up) * np.log(1-p_up)
    
    int_row = [f_r3, f_dt, f_sigma, f_dsigma, f_natr, f_rv, f_intensity, f_skew, f_entropy]
    
    ext_row = []
    if ext_tickers:
        for t in ext_tickers:
            if t in df_d.columns:
                ext_row.append((df_d[t].iloc[-1] - df_d[t].iloc[-int(best_N)]) / (df_d[t].iloc[-int(best_N)] + 1e-9))
            else:
                ext_row.append(0.0)

    features = int_f + ext_f
    sC = StandardScaler()
    Cti = sC.fit_transform(final_df[features]) if features else np.empty((len(final_df), 0))
    Yt = final_df["y"].values
    
    train_x = torch.tensor(Cti, dtype=torch.float32)
    train_y = torch.tensor(Yt, dtype=torch.float32)
    
    model, likelihood = train_dkl(train_x, train_y)
    model.eval()
    likelihood.eval()

    test_row = int_row + ext_row
    Csi = sC.transform(np.array([test_row])) if features else np.empty((1,0))
    test_x = torch.tensor(Csi, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mu_f = pred.mean.item()
        std_f = pred.stddev.item()

    return {
        'Stock': ticker,
        'Sectors': sector_codes,
        'Opt. N': int(best_N),
        'Eff. Index': f"{best_score:.2f}",
        'Hit Rate': f"{best_stats[0]:.1f}%",
        'Pred Mean (Z)': f"{mu_f:+.2f}",
        '95% CI (±Z)': f"{1.96 * std_f:.2f}",
        'Signal': 'UP' if mu_f > 0 else 'DOWN'
    }

# ==========================================
# 3. Main Scanner Execution
# ==========================================
if __name__ == "__main__":
    print(f"==========================================")
    print(f"DEEP GP (DKL) PORTFOLIO SCANNER")
    print(f"==========================================")
    
    # Process portfolio in parallel (Stock level)
    results = Parallel(n_jobs=-1, backend="threading")(delayed(process_single_stock)(t, s) for t, s in PORTFOLIO_INPUT)
    
    clean_results = [r for r in results if r is not None]
    if clean_results:
        df_portfolio = pd.DataFrame(clean_results)
        print("\n\nFINAL DEEP GP PORTFOLIO INTELLIGENCE TABLE:")
        print(df_portfolio.to_string(index=False))
        df_portfolio.to_csv('portfolio_deepgp_results.csv', index=False)
        print(f"\nResults saved to portfolio_deepgp_results.csv")
    else:
        print("No results generated.")
