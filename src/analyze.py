import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


file_path = 'FordM.csv' 
monthly_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)


split_point = len(monthly_data) - 12  # Last 12 months as test set
train, test = monthly_data[0:split_point], monthly_data[split_point:]


auto_model = auto_arima(train['Sales'], seasonal=True, m=12,
                        start_p=1, start_q=1,
                        max_p=3, max_q=3, max_d=2,
                        start_P=1, start_Q=1, max_P=3, max_Q=3, max_D=2,
                        trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

print(auto_model.summary())


model = SARIMAX(train['Sales'], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
model_fit = model.fit()


forecast = model_fit.forecast(steps=len(test))


r2 = r2_score(test['Sales'], forecast)
print("R-squared: ", r2)

plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Sales'], label='Train Sales')
plt.plot(test.index, test['Sales'], label='Test Sales')
plt.plot(test.index, forecast, label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Sales Forecast vs Actual')
plt.legend()
plt.show()
