import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_days = 365
dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=num_days))
sales = 100 + 50 * np.sin(2 * np.pi * np.arange(num_days) / 30) + np.random.normal(0, 20, num_days) #Seasonal sales with noise
temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(num_days) / 365) + np.random.normal(0, 5, num_days) #Seasonal temperature with noise
promotions = np.random.choice([0, 1], size=num_days, p=[0.9, 0.1]) # 10% chance of promotion each day
df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Temperature': temperature, 'Promotion': promotions})
# --- 2. Data Cleaning and Feature Engineering ---
df['DayOfYear'] = df['Date'].dt.dayofyear
df['DayOfWeek'] = df['Date'].dt.dayofweek #Monday=0, Sunday=6
df['IsWeekend'] = df['DayOfWeek'] >=5
df['Month'] = df['Date'].dt.month
# --- 3. Analysis and Modeling ---
#Simple Linear Regression to predict sales based on temperature and promotions
X = df[['Temperature', 'Promotion', 'DayOfYear', 'IsWeekend']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
df['PredictedSales'] = predictions
#Evaluate the model (example using R-squared)
r_squared = model.score(X,y)
print(f"R-squared: {r_squared}")
# --- 4. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Actual Sales')
plt.plot(df['Date'], df['PredictedSales'], label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = 'sales_prediction.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
sns.regplot(x='Temperature', y='Sales', data=df)
plt.title('Sales vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Sales')
output_filename2 = "sales_temp.png"
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")