import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

file_path = 'FordM.csv'  
monthly_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
data = pd.read_csv('FordM.csv')
# Function to create lagged features
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

# Splitting the data into features (X) and target (y)
X = monthly_data_lagged.drop('Sales', axis=1)
y = monthly_data_lagged['Sales']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
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

# Using Feature Importance for Feature Selection
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Selecting top N important features
N = 10  # Number of top features to select, can be tuned
X_train_selected = X_train[:, indices[:N]]
X_test_selected = X_test[:, indices[:N]]

# Trying Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_selected, y_train)
y_pred_gb = gb_model.predict(X_test_selected)

# Define the MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluating the Gradient Boosting model with MAPE
mse = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)
accuracy_percentage_gb = 100 - mape_gb
print("Gradient Boosting - Root Mean Squared Error:", rmse_gb)
print("Gradient Boosting - Mean Squared Error:", mse)
print("Gradient Boosting - R-squared:", r2_gb)
print("Gradient Boosting - Model Accuracy:", accuracy_percentage_gb, '%')

#data['Forecast'] = model.predict( end=len(data), typ='levels')

dates = monthly_data_lagged.index


train_size = int(len(X_scaled) * 0.8)
train_dates = dates[:train_size]
test_dates = dates[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label='Actual Sales', color='blue')
plt.plot(test_dates, y_pred_gb, label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales - Gradient Boosting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.legend()
plt.tight_layout()  # Adjust layout for better fit
plt.show()

y_pred_train_gb = gb_model.predict(X_train_selected)


combined_predictions = np.concatenate((y_pred_train_gb, y_pred_gb), axis=0)

# Combine train and test dates for plotting
combined_dates = np.concatenate((train_dates, test_dates), axis=0)

# Plotting the results for the entire dataset
plt.figure(figsize=(12, 6))
plt.plot(combined_dates, monthly_data_lagged['Sales'].values, label='Actual Sales', color='blue')
plt.plot(combined_dates, combined_predictions, label='Model Predictions', color='red')
plt.title('Actual vs Model Predictions - Gradient Boosting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label='Actual Sales', color='blue')
plt.plot(test_dates, y_pred_gb, label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales - Gradient Boosting')
plt.xlabel('Date')
plt.ylabel('Sales')

# Formatting the x-axis to display months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.show()
