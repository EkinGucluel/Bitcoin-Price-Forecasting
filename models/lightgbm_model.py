# models/lightgbm_model.py

from .base_model import BaseModel
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

class LightGBMModel(BaseModel):
    """
    LightGBM Regressor model.
    """
    def __init__(self, params=None):
        if params is None:
            # Starter parameters
            params = {'objective': 'regression_l1', 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
        super().__init__("LightGBM", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name} with params: {self.params}")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def tune(self, X_train, y_train):
        print(f"--- Tuning Hyperparameters for {self.model_name} ---")

        # Define the grid of parameters to search
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40, 50],          # Controls complexity, main parameter to tune
            'max_depth': [-1, 10, 20],               # -1 means no limit
            'reg_alpha': [0.1, 0.5, 1.0],            # L1 regularization
            'reg_lambda': [0.1, 0.5, 1.0],           # L2 regularization
            'colsample_bytree': [0.7, 0.8, 0.9]      # Subsampling of columns
        }

        # Set up the RandomizedSearchCV
        tuner = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1),
            param_distributions=param_grid,
            n_iter=25,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=2,
            random_state=42
        )

        # Run the search
        tuner.fit(X_train, y_train)

        # Save the best parameters found
        print(f"Best parameters found: {tuner.best_params_}")
        self.params = tuner.best_params_

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)