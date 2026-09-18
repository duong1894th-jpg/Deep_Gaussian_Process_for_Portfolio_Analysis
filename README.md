# Vietnam Banking Deep Gaussian Process (DeepGP)

This repository contains a quantitative financial model utilizing **Deep Kernel Learning (DKL)** to forecast the stock price movements of the top 10 Vietnamese banks.

By leveraging a PyTorch Neural Network feature extractor followed by an Exact Gaussian Process (`gpytorch`), the model dynamically learns to blend both internal price-action metrics and external macroeconomic context (e.g., USD/VND exchange rates, U.S. Treasury yields) to predict market signals.

## Algorithm Overview
This project uses **Deep Kernel Learning (DKL)** to combine the representation-learning power of deep neural networks with the reliable uncertainty quantification of Gaussian Processes (GPs).

1. **Context Feature Extractor**: Both internal metrics (price momentum, volatility, volume intensity, etc.) and external macroeconomic signals (bond yields, exchange rates, commodity futures) are fed into a PyTorch Neural Network. The network projects this complex, raw data into a dense, lower-dimensional embedding.
2. **Exact Gaussian Process**: Instead of computing a standard RBF kernel on the raw data, the GP computes the kernel on the *neural network's embeddings*. This allows the model to naturally capture complex, non-linear dependencies between the macro environment and the stock's price action.
3. **Joint Training**: The neural network weights and the GP hyperparameters (lengthscales, noise variances) are jointly optimized end-to-end via gradient descent to maximize the Exact Marginal Log Likelihood.

## Evaluation Pipeline
How the model evaluates each stock:
1. **Time-Series Chunking**: Historical data is sliced into discrete $N$-day blocks. The model extracts internal price-action behaviors (volatility, momentum, volume intensity, skewness) and context shifts over a block to predict the target return of the subsequent block.
2. **Backtesting & Optimization**: The algorithm tests multiple time horizons ($N$-day lookbacks) over the most recent `HORIZON_BLOCKS` (e.g., the last 10 chunks). It selects the optimal $N$ that maximizes the historical "Hit Rate" (directional accuracy) relative to the prediction errors (RMSE).
3. **Future Forecasting**: Once the optimal time horizon is identified, the model ingests the most recent $N$-day market data and projects the expected return (Mean Z-score) and confidence interval for the upcoming period, generating a clear `UP` or `DOWN` signal.

## Features
- **Deep Kernel Learning**: Neural network feature extraction fused with a Gaussian Process.
- **Automated Context Integration**: Automatically evaluates external macro conditions alongside internal stock behaviors.
- **Top 10 Banks Configured**: Pre-configured to scan VCB, BID, CTG, TCB, MBB, VPB, ACB, STB, HDB, and VIB.
- **Sector Mapping Reference**: Includes `vietnam_stock_sectors_external_features.csv` as a master lookup guide to easily run the code for *any* stock based on its sector and relevant macro features.

## Installation
Ensure you have Python installed, then install the required AI libraries:
```bash
pip install torch gpytorch yfinance pandas numpy scikit-learn joblib
```

## Usage
Simply run the scanner script:
```bash
python portfolio_DeepGP.py
```
This will fetch the latest market data, train the Deep GP network on the fly for each bank, and output the projected signals (UP/DOWN) along with confidence intervals.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
