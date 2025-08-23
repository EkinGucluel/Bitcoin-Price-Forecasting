# Bitcoin-Price-Forecasting

This project provides a framework for forecasting the price of Bitcoin using a diverse range of models. It uses historical Bitcoin data, processes it into a daily dataset, and applies various classical, machine learning, and deep learning models to predict future prices. 

The project is structured with an object-oriented approach, making it easy to add, test, and compare new models. It serves as a practical, in-depth exploration of time-series analysis and forecasting in the highly volatile cryptocurrency market.

# Core Python Modules Used:
- pandas for data manipulation and analysis.
- NumPy for numerical operations.
- scikit-learn for machine learning models (Random Forest), hyperparameter tuning (RandomizedSearchCV), and evaluation metrics.
- XGBoost & LightGBM for high-performance gradient boosting models.
- statsmodels & pmdarima for classical time-series models (ARIMA/SARIMA).
- Prophet (by Meta) for robust, decomposable time-series forecasting.
- TensorFlow with the Keras API for building and training deep learning models (LSTM, GRU).
- Plotly for creating interactive and insightful visualizations.
- pandas-ta for generating a wide range of technical analysis indicators.
# The project provides:
- Data Preprocessing: A pipeline that cleans and resamples minute-level data into a daily format.
- Extensive Feature Engineering: Generation of over 20 features, including:
  - Price-based: Percentage and Log Returns.
  - Technical Indicators: RSI, MACD, Bollinger Bands, ATR.
  - Volume-based: On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP).
  - Time-based: Lag features (1-day, 7-day, 30-day) and Rolling Statistics (7-day, 30-day).
- Diverse Model Implementation: A portfolio of 15+ individual models, including:
  - Classical: ARIMA, SARIMA.
  - Specialized: Prophet.
  - Machine Learning: Random Forest, XGBoost, LightGBM.
  - Deep Learning: Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, and GRU with varying sequence lengths.
- Automated Hyperparameter Tuning: Systematic search for the best parameters for each model to maximize performance.
- Ensemble Modeling: Implementation of Simple Averaging and Stacking with a meta-learner to combine the predictions of the top-performing models.
- Evaluation: Comparison of all models using standard regression metrics (RMSE, MAE).
- Visualizations: A suite of interactive plots to analyze the data and results, including:
  - Actual vs. Predicted Forecast Plots (separated by model family).
  - Residual (Error) Plots.
  - Feature Importance Charts for tree-based models.
  - Training & Validation Loss Curves for deep learning models.
  - Prediction Interval Plots for Prophet.
# Structure
The project is architected in an object-oriented and modular fashion to promote reusability and ease of experimentation.
1. Data Preprocessing (data_preprocessing.ipynb):
This notebook is the starting point. It loads the raw minute-level data from Kaggle.
It handles data cleaning, resampling to a daily frequency, and generates all the engineered features.
Its final output is a set of cleaned and split data files (X_train.pkl, y_train.pkl, etc.) that are ready for model training.
2. Model Training and Analysis (main.ipynb):
This is the main script that orchestrates the entire experiment.
It loads the preprocessed data.
It initializes instances of all the different models.
It contains a master loop that trains, tunes (optional), predicts, and evaluates each model.
Finally, it runs the ensemble methods and generates all the comparison charts and tables.
3. Modular Model Classes (models/ directory):
base_model.py: A parent "BaseModel" class defines a standard interface (.train(), .predict(), .tune(), .evaluate()) that all other model classes must follow.
Individual Model Files: Each model (e.g., xgboost_model.py, lstm_model.py) is implemented in its own file as a class that inherits from BaseModel. This makes the code clean, self-contained, and easy to manage and extend.

# Installation
Clone the repository:
- 'git clone https://github.com/your-username/your-repository-name.git'
'cd your-repository-name'
Create a virtual environment (recommended):
- 'python -m venv myenv'
- 'source myenv/bin/activate'  # On Windows, use `myenv\Scripts\activate`
### Install the required modules:
- 'pip install -r requirements.txt'
(Note: On macOS, you may need to install the libomp library for XGBoost to work: brew install libomp)

# How to use:
- Download the Data: Download the Bitcoin historical data from this Kaggle dataset and place the .csv file in the data/ directory.
- Run the Preprocessing Notebook: Open and run the data_preprocessing.ipynb notebook from top to bottom. This will generate the processed_data/ folder with the necessary .pkl files.
- Run the Main Experiment Notebook: Open and run the main.ipynb notebook. This will train all the models, run the ensembles, and generate all the final results, tables, and visualizations. You can use the PERFORM_TUNING switch at the top of the notebook to control whether to run the time-consuming hyperparameter tuning process.
# Results & Visualizations
### Example: Machine Learning Forecasts vs. Actual Price
<img width="1160" height="450" alt="ML_plot_after_hypertuning" src="https://github.com/user-attachments/assets/73804790-7368-4e80-83a7-e28c4c7864d6" />

### Example: Deep Learning Forecasts vs. Actual Price
<img width="1160" height="450" alt="newplot" src="https://github.com/user-attachments/assets/166da29c-c4da-4396-a8bc-3b1ede862ca1" />

### Example: Ensemble Model Performance
<img width="1160" height="450" alt="Ensemble Approach" src="https://github.com/user-attachments/assets/534f7661-8c2a-4799-a32f-06bd06cc539e" />

### Example: Final Model Comparison
<img width="498" height="375" alt="Screenshot 2025-08-23 at 8 23 23 PM" src="https://github.com/user-attachments/assets/e2fff522-5191-4f81-9c2c-d45827509449" />


## If any library clashes occur:

A small difference in pandas-ta is required to resolve a small clashing.
Go into `myenv/lib/python3.12/site-packages/pandas_ta/momentum/squeeze_pro.py`
Replace `from numpy import NaN as npNaN` with `import numpy as np npNaN = np.nan`
