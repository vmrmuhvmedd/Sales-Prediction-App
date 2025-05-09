import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

df = pd.read_csv('output.csv')

X = df.drop(['Sales', 'Product Name'], axis=1)
X = pd.get_dummies(X, drop_first=True)  
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=0.001),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'XGBoost': XGBRegressor(random_state=42),
    'Voting Regressor': VotingRegressor(estimators=[
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('rfr', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(random_state=42))
    ])
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {
        'model': model,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': rmse
    }

best_model_name = min(results, key=lambda name: results[name]['RMSE'])
best_model = results[best_model_name]['model']

print(f"\n✅ Best Model: {best_model_name}")
print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

features_columns = X.columns.tolist()
with open('features_columns.pkl', 'wb') as f:
    pickle.dump(features_columns, f)