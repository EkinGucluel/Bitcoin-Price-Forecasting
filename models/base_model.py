# models/base_model.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class BaseModel:
    """
    Abstract base class for all models.
    """
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """Trains the model. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict(self, X_test):
        """Makes predictions. Must be implemented by subclasses."""
        raise NotImplementedError

    def tune(self, X_train, y_train):
        """Performs hyperparameter tuning. Optional to implement."""
        print(f"Tuning not implemented for {self.model_name}")
        return self.params # Return default params

    def evaluate(self, y_true, y_pred):
        """Calculates and returns evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"--- {self.model_name} Metrics ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        return {'rmse': rmse, 'mae': mae}