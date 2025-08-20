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
            # These are now just starter params, tuning will find better ones
            params = {'objective': 'reg:squarederror', 'n_jobs': -1}
        super().__init__("XGBoost", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name} with params: {self.params}")
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, verbose=False)
        print("Training complete.")


    def tune(self, X_train, y_train):
        print(f"--- Tuning Hyperparameters for {self.model_name} ---")

        # 1. Define the grid of parameters to search
        param_grid = {
            'n_estimators': [100, 500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        # Set up the RandomizedSearchCV
        # n_iter=25 means it will try 25 different random combinations.
        # cv=3 means it will use 3-fold cross-validation.
        # scoring='neg_root_mean_squared_error' is what we want to optimize.
        tuner = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1),
            param_distributions=param_grid,
            n_iter=25,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=2, # Will print progress
            random_state=42
        )

        # Run the search
        tuner.fit(X_train, y_train)

        # Save the best parameters found
        print(f"Best parameters found: {tuner.best_params_}")
        self.params = tuner.best_params_

        # self.model = tuner.best_estimator_

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)