from .base_model import BaseModel
from prophet import Prophet

class ProphetModel(BaseModel):
    """
    Prophet model
    """
    def __init__(self, params=None):
        if params is None:
            params={} # Prophet has many tuning parameters, start with defaults
        super().__init__("Prophet", params)

    def train(self, x_train, y_train):
        print(f"Training {self.model_name}...")

        # Prophet requires a specific DataFrame format: ['ds', 'y']
        prophet_df = y_train.reset_index()
        prophet_df.columns = ['ds', 'y']

        # 2. Initialize and fit the model
        self.model = Prophet(**self.params)
        self.model.fit(prophet_df)
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")

        # 1. Create a "future" DataFrame for the dates we want to predict
        n_periods = len(X_test)
        future_df = self.model.make_future_dataframe(periods=n_periods, freq='D')

        # 2. Make the prediction
        forecast_df = self.model.predict(future_df)

        # 3. Extract just the prediction ('yhat') for the test period
        # The forecast_df contains predictions for both training and test periods
        predictions = forecast_df['yhat'][-n_periods:]

        return predictions.values # Return as a numpy array for consistency