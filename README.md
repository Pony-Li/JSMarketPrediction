# Jane Street Real-Time Market Data Forecasting

This repository contains the **bronze-winning solution (ranked 321/3757)** for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition on Kaggle.

The project includes:

- Data analysis
- Feature engineering and categorical encoding
- Convolutional GRU-based time series modeling (offline training)
- Online incremental learning and inference

---

## 📁 Project Structure

```text
├── data/                          # All training & test datasets
│   ├── original_test.parquet      # Official simulated test data
│   ├── original_lags.parquet      # Simulated lag features provided by organizers
│   ├── train.parquet              # Initial full training data
│   ├── test.parquet               # Simulated test data synthesized from train.parquet (for online learning)
│   ├── lags.parquet               # Simulated lag features (for online learning)
│   ├── train_data.parquet         # Filtered training data (date_id > 1100)
│   └── valid_data.parquet         # Filtered validation data (date_id > 1640)

├── kaggle_evaluation/            # Scripts to simulate online prediction via official interface

├── model/
│   └── GRU.ckpt                   # Checkpoint of GRU model after offline training

├── notebooks/
│   ├── EDA.ipynb                  # Exploratory data analysis
│   └── feature-engineering.ipynb  # Attempts at feature engineering

├── scripts/
│   └── run_all.sh                 # One-click script for data generation, training, and inference

├── src/
│   ├── synthetic_data.py          # Generate synthetic data for training
│   ├── train.py                   # Training entry point, includes validation logic
│   ├── inference.py               # Local inference entry point, starts inference server
│   ├── model_gru.py               # GRU model architecture and parameters
│   ├── online_predictor.py        # JsGruOnlinePredictor class for online inference + learning
│   ├── utils.py                   # Utility functions: encoding, R², etc.
│   └── __init__.py                # Marks src as a Python module

├── requirements.txt              # Python dependencies
├── setup.py                      # Project installation script
├── submission.parquet            # Submission file generated after running inference
└── README.md                     # This file

## 🚀 Getting Started

Clone the repository and set up the environment as follows:

```bash
# Clone the repository
git clone https://github.com/<your-username>/JSMarketPrediction.git
cd JSMarketPrediction

# (Optional) Create a virtual environment
conda create -n jsmarket python=3.10
conda activate jsmarket

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
bash scripts/run_all.sh
