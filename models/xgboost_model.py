# models/xgboost_model.py

from .base_model import BaseModel
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

class XgboostModel(BaseModel):
    """
    XGBoost Regressor model.
    """
    def __init__(self, params=None):
        # Set default parameters if none are provided
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1
            }
        super().__init__("XGBoost", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name}...")
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, verbose=False)
        print("Training complete.")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)

    # You can add a specific `tune` method here later using GridSearchCV or RandomizedSearchCV