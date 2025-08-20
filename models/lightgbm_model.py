# models/lightgbm_model.py

from .base_model import BaseModel
import lightgbm as lgb

class LightGBMModel(BaseModel):
    """
    LightGBM Regressor model.
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'regression_l1', # MAE
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1 # Suppress verbose output
            }
        super().__init__("LightGBM", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name}...")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)