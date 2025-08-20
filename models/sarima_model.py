from .base_model import BaseModel
import pmdarima as pm

class SarimaModel(BaseModel):
    """
    SARIMA model that automatically finds the best seasonal and non-seasonal parameters.
    """
    def __init__(self, params=None):
        # The 'm' parameter defines the seasonal period. For daily data, m=7 is weekly seasonality.
        if params is None:
            params = {'m': 7}
        super().__init__("SARIMA", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name} by finding the best parameters (m={self.params['m']})...")

        y_train_with_freq = y_train.asfreq('D').ffill()

        # The key difference from ARIMA is seasonal=True and the 'm' parameter.
        auto_model = pm.auto_arima(
            y_train_with_freq,
            start_p=1, start_q=1,
            test='adf',
            max_p=3, max_q=3,
            m=self.params['m'], # Set the seasonal period
            start_P=0,
            seasonal=True,     # Enable seasonal search
            d=None, D=None,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        self.model = auto_model

        print("\n--- Auto-SARIMA Found Best Model ---")
        print(self.model.summary())
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")

        n_periods = len(X_test)
        return self.model.predict(n_periods=n_periods)