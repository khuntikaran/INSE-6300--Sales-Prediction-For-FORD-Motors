import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = pd.read_csv('FordM.csv')
model = SARIMAX(data['Sales'],
                order=(1, 1, 1),             
                seasonal_order=(1, 1, 1, 12),
                # exog=exog                  
               )


model_fit = model.fit()


print(model_fit.summary())


data['Forecast'] = model_fit.predict(start=1, end=len(data), typ='levels')


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
