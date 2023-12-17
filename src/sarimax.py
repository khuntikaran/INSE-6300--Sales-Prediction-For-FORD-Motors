import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset
data = pd.read_csv('FordM.csv')

# Assuming 'Sales' is your target variable and 'data' is your DataFrame
# Replace 'exog_variable' with actual external variables if you have any
# exog = data['exog_variable']

# Define the SARIMAX model
# Modify these parameters according to your dataset's seasonality
model = SARIMAX(data['Sales'],
                order=(1, 1, 1),              # These are the non-seasonal ARIMA parameters: (p, d, q)
                seasonal_order=(1, 1, 1, 12), # These are the seasonal ARIMA parameters: (P, D, Q, S)
                # exog=exog                   # Uncomment if you have external variables
               )

# Fit the model
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecasting
data['Forecast'] = model_fit.predict(start=1, end=len(data), typ='levels')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data['Sales'], label='Actual Sales')
plt.plot(data['Forecast'], color='red', label='Forecasted Sales')
plt.title('Sales Forecast using SARIMAX')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Accuracy Metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(data['Sales'][1:], data['Forecast'][1:])
accuracy_percentage = 100 - mape
mse = mean_squared_error(data['Sales'][1:], data['Forecast'][1:])
rmse = np.sqrt(mse)

print("MAPE (Accuracy Percentage): {:.2f}%".format(accuracy_percentage))
print("Mean Squared Error: {:.2f}".format(mse))
print("Root Mean Squared Error: {:.2f}".format(rmse))
