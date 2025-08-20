# models/random_forest_model.py

from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

class RandomForestModel(BaseModel):
    """
    Random Forest Regressor model.
    """
    def __init__(self, params=None):
        if params is None:
            # Starter parameters
            params = {'random_state': 42, 'n_jobs': -1}
        super().__init__("Random Forest", params)

    def train(self, X_train, y_train):
        print(f"Training {self.model_name} with params: {self.params}")
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def tune(self, X_train, y_train):
        print(f"--- Tuning Hyperparameters for {self.model_name} ---")

        # Define the grid of parameters to search
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],        # Number of trees in the forest
            'max_depth': [10, 20, 30, None],              # Maximum depth of the tree (None means no limit)
            'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required at a leaf node
            'max_features': ['sqrt', 'log2', 1.0]           # Number of features to consider when looking for the best split
        }

        # Set up the RandomizedSearchCV
        tuner = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=25, # Try 25 different combinations
            cv=3,      # Use 3-fold cross-validation
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