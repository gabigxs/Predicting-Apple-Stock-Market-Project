import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('AppleStockMarket.csv')

# Convert 'Date' to datetime format and sort the dataframe by date
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values(by='Date')

# Create a new column 'Days' that counts the number of days from the start of the dataset
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Define features and target
X = df['Days'].values   # We use 'Days' as our feature
y = df['Close'].values  # 'Close' price is the target

# Calculate the mean of X and Y
mean_x = np.mean(X)
mean_y = np.mean(y)

# Calculating the slope (m) and intercept (b) using the least squares method
numerator = 0
denominator = 0
for i in range(len(X)):
    numerator += (X[i] - mean_x) * (y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
m = numerator / denominator
b = mean_y - m * mean_x

# Function to predict future stock prices in months
def predict_future_prices_in_months(start_month, end_month):
    start_date = datetime(2010, 1, 1) + timedelta(days=30 * (start_month - 1))
    end_date = datetime(2010, 1, 1) + timedelta(days=30 * (end_month - 1))

    start_days = (start_date - df['Date'].min()).days
    end_days = (end_date - df['Date'].min()).days
    
    future_days = np.arange(start_days, end_days + 1)
    future_prices = m * future_days + b
    
    predictions = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date),
        'Predicted_Close': future_prices
    })
    
    return predictions

# User input for the number of months for prediction
start_month = 171
additional_months = int(input("Number of months for prediction: "))
end_month = start_month + additional_months

# Predict and plot the prices for the custom period
predicted_prices = predict_future_prices_in_months(start_month, end_month)

# Plotting the predictions for the custom period
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices['Date'], predicted_prices['Predicted_Close'], color='blue', label='Predicted Close Prices')
plt.title('Predicted Apple Stock Prices for Custom Period')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('Predicted_Apple_Stock_Prices_Custom_Period.png')

# Show the plot
plt.show()

# Save the predictions to a CSV file
predicted_prices.to_csv('Predicted_Apple_Stock_Prices.csv', index=False)

print("Plot saved as 'Predicted_Apple_Stock_Prices_Custom_Period.png'")
print("Predictions saved to 'Predicted_Apple_Stock_Prices.csv'")
