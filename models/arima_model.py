# models/arima_model.py

from .base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA

class ArimaModel(BaseModel):
    """
    ARIMA model.
    """
    def __init__(self, params=None):
        if params is None:
            # (p, d, q) order
            params = {'order': (5, 1, 0)}
        super().__init__("ARIMA", params)

    # ARIMA's train method is different, it only needs the target series (y_train)
    def train(self, X_train, y_train):
        print(f"Training {self.model_name}...")
        y_train_with_freq = y_train.asfreq('D')
        self.model = ARIMA(y_train_with_freq, order=self.params['order']).fit()
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        # Predict for the number of steps in the test set
        n_periods = len(X_test)
        return self.model.forecast(steps=n_periods)