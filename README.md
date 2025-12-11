# Physics-Guided Bi-LSTM for Earthquake Forecasting (Zagros-Makran)

This repository contains the source code and implementation details for the paper:
**"Physics-Guided Bi-LSTM Network with Uncertainty Quantification for Earthquake Magnitude Forecasting in the Zagros-Makran Transition Zone"**

## Repository Contents

* `main_forecast.py`: The primary script. It handles data loading, feature engineering (b-value, energy), model training (ARIMA, RF, Vanilla LSTM, Bi-LSTM), and generates all statistical tables and figures used in the manuscript.
* `run_quick_test.py`: A lightweight script to verify the model architecture and environment without needing the full dataset.
* `requirements.txt`: List of required Python libraries.
* `Final_Cleaned_Catalog_v2.csv`: (Optional) The pre-processed seismic catalog file.

## Usage Instructions

### 1. Installation
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt