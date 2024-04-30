import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from data_loader import DataLoader

class RandomForestReg:
    def __init__(self, input_keys, target_key):
        self.data_loader = DataLoader()
        self.target_key = target_key
        self.input_keys = self.data_loader.input_keys
        self.data = self.data_loader.data
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.X_y(target_key=self.target_key)
        self.model()
        self.error()
        self.full_model()

    def model(self):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        # Instantiate Random Forest regressor
        self.model = RandomForestRegressor(random_state=42)

        # Perform Grid Search to find best hyperparameters
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Set the model with the best hyperparameters
        self.model = grid_search.best_estimator_

    def error(self):
        prediction = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, prediction)
        mse = mean_squared_error(self.y_test, prediction)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)

    def full_model(self):
        X = self.data[self.input_keys].values
        y = self.data[self.target_key].values
        self.model = RandomForestRegressor()
        self.model.fit(X, y)
        joblib.dump(self.model, f"./models/{self.target_key}_model.joblib")
