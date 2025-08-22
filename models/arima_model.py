from .base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

class ArimaModel(BaseModel):
    """
    ARIMA model.
    """
    def __init__(self, params=None):
        # if params is None:
        #     # (p, d, q) order
        #     params = {'order': (5, 1, 0)}
        super().__init__("ARIMA", params)

    # ARIMA's train method is different, it only needs the target series (y_train)
    def train(self, X_train, y_train):
        print(f"Training {self.model_name} by finding the best parameters...")
        y_train_with_freq = y_train.asfreq('D') # Ensures date index is correct

        y_train_with_freq = y_train_with_freq.ffill()
        # The .fit() is done inside the pm.auto_arima()
        # Test different parameters and finds the best one that fits the data.
        auto_model = pm.auto_arima(
            y_train_with_freq,
            start_p=1, start_q=1,
             test='adf',
             max_p=5, max_q=5,
             m=1,
             d=None,
             seasonal=False,
             start_P=0,
             D=0,
             trace=True, # This will print out the models it is testing.
             error_action='ignore',
             suppress_warnings=True,
             stepwise=True
        )

        self.model = auto_model
        print("\n--- Auto-ARIMA Found Best Model ---")
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        # Predict for the number of steps in the test set
        n_periods = len(X_test)
        return self.model.predict(n_periods=n_periods)