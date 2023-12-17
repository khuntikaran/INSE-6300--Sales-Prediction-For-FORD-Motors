import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


file_path = 'FordM.csv'  
monthly_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)


def create_lagged_features(df, n_lags=12):
    df_lagged = df.copy()
    for i in range(1, n_lags + 1):
        df_lagged[f'lag_{i}'] = df_lagged['Sales'].shift(i)
    return df_lagged

# Adding rolling statistics as new features
monthly_data_lagged = create_lagged_features(monthly_data)
monthly_data_lagged['rolling_mean_3'] = monthly_data_lagged['Sales'].rolling(window=3).mean()
monthly_data_lagged['rolling_std_3'] = monthly_data_lagged['Sales'].rolling(window=3).std()
monthly_data_lagged.dropna(inplace=True)  # Dropping NaN values created by rolling windows

X = monthly_data_lagged.drop('Sales', axis=1)
y = monthly_data_lagged['Sales']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Randomized Search for Hyperparameter tuning
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                   param_distributions=param_distributions,
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Best parameters and retraining
best_params = random_search.best_params_
print("Best Parameters:", best_params)

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluating the Gradient Boosting model with MAPE
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
r2_gb = r2_score(y_test, y_pred)
mape_gb = mean_absolute_percentage_error(y_test, y_pred)
accuracy_percentage_gb = 100 - mape_gb
print("Root Mean Squared Error:", rmse_gb)
print("Model Accuracy:", accuracy_percentage_gb, '%')

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Sales', color='blue')
plt.plot(y_pred, label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales - Gradient Boosting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Scatter Plot of Actual vs Predicted Sales')
plt.grid(True)
plt.show()
