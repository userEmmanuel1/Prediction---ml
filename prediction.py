import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score  # Import precision_score

##### INPUT TRACKING STOCK
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")

# Remove unnecessary columns from the DataFrame
sp500_data = sp500_data.drop(columns=["Dividends", "Stock Splits"])

### GRAPH, LEARN TO PRINT ON PAGE ######
plt.plot(sp500_data.index, sp500_data['Close'], label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('S&P 500 Closing Prices Over Time')
plt.legend()
#plt.show()

# Create a new column "Tomorrow" with closing prices shifted by one day
# Create a new column "Tomorrow" with closing prices shifted by one day
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)

# Adding a binary target variable
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)

# Assuming "Target" is the column you want to predict
train = sp500_data.iloc[:-100]
test = sp500_data.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

# Make predictions on the test set
preds = model.predict(test[predictors])

precision = precision_score(test["Target"], preds)
print("Precision Score:", precision)

# Create a DataFrame to combine true and predicted values
combine = pd.concat([test["Target"], pd.Series(preds, index=test.index, name="Predicted")], axis=1)
combine.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Index')
plt.ylabel('Target and Predicted Values')
plt.title('Comparison of Target and Predicted Values')
plt.legend(["Target", "Predicted"])
plt.show()