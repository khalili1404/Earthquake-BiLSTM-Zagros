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
```

### 2. Quick Test (For Reviewers)
To verify that the code environment is set up correctly and the model architecture is valid, run the quick test script. This does not require the dataset.

```bash
python run_quick_test.py
```

Expected Output: "QUICK TEST PASSED SUCCESSFULLY"

### 3. Reproducing Paper Results

To reproduce the training process, statistical analysis (Table 1), and figures (Figs 1-7):

Place your dataset file named Final_Cleaned_Catalog_v2.csv in the root directory.

Run the main script:

```bash
python main_forecast.py
```

## Model Overview
The proposed model utilizes a Bi-Directional LSTM with a Self-Attention Mechanism. It integrates physics-based features:

Log-Energy: Derived from Gutenberg-Richter relation.

b-value: Calculated on a rolling window to capture stress states.

Uncertainty Quantification: Implemented via Monte Carlo Dropout to estimate epistemic uncertainty.

Citation
If you use this code or dataset in your research, please cite:

```
@article{khalili2025hierarchical,
  title={Physics-Guided Bi-LSTM Network with Uncertainty Quantification for Earthquake Magnitude Forecasting in the Zagros-Makran Transition Zone},
  author={Khalili, Marzieh and Fotoohi, Ali},
  journal={Computers & Geosciences (Under Review)},
  year={2025}
}
```

Contact
For questions or inquiries, please contact: Marzieh Khalili - marzieh-khalili@shirazu.ac.ir

