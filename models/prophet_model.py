from .base_model import BaseModel
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import pandas as pd

# The forecast is simply the sum of these components: Forecast = Trend + Seasonality.
class ProphetModel(BaseModel):
    """
    Prophet model
    """
    def __init__(self, params=None):
        if params is None:
            params={} # Prophet has many tuning parameters, start with defaults
        super().__init__("Prophet", params)

    def train(self, x_train, y_train):
        print(f"Training {self.model_name} with params: {self.params}")

        # Prophet requires a specific DataFrame format: ['ds', 'y']
        prophet_df = y_train.reset_index()
        prophet_df.columns = ['ds', 'y'] # ds for datestamp

        # Initialize and fit the model
        self.model = Prophet(**self.params) # (**) → unpacks a dict into keyword arguments.
        self.model.fit(prophet_df) # Trains on the dataframe
        print("Training complete.")

    def tune(self, X_train, y_train):
        print(f"--- Tuning Hyperparameters for {self.model_name} ---")

        # Prepare the data in the format Prophet needs
        prophet_df = y_train.reset_index()
        prophet_df.columns = ['ds', 'y']

        # Define the grid of parameters to search
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], # How flexible the trend is
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0], # How flexible the seasonality is
        }

        # Create all combinations of parameters for a Grid Search
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        rmses = [] # List to store the RMSE for each param combination

        # Manually loop through the parameter combinations
        for params in all_params:
            print(f"Testing params: {params}")

            # Initialize and train the model with the current set of parameters
            m = Prophet(**params).fit(prophet_df)

            # Perform cross-validation
            # initial='730 days' means the first training period is 2 years
            # period='180 days' means we make a new forecast every 6 months
            # horizon='365 days' means we forecast 1 year into the future
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")

            # Get performance metrics
            df_p = performance_metrics(df_cv, rolling_window=1)

            # Store the final RMSE
            rmses.append(df_p['rmse'].values[0])

        # Find the best parameters
        best_params_index = rmses.index(min(rmses))
        best_params = all_params[best_params_index]

        print(f"\nBest parameters found: {best_params} with RMSE: {min(rmses)}")
        self.params = best_params

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")

        # Create a "future" DataFrame for the dates we want to predict
        n_periods = len(X_test)
        future_df = self.model.make_future_dataframe(periods=n_periods, freq='D')

        # Make the prediction
        forecast_df = self.model.predict(future_df)

        # Extract just the prediction ('yhat') for the test period
        # The forecast_df contains predictions for both training and test periods
        predictions = forecast_df['yhat'][-n_periods:]

        return predictions.values # Return as a numpy array for consistency