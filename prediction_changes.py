import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score  # Import precision_score
import numpy as np


##### INPUT TRACKING STOCK
sp500 = yf.Ticker("^GSPC")
#Reduced time period because things such as a 1987 Black Monday, 2000 Dot-Com Bubble, 2008 Housing Crisis are noise
#2018 crash is also noise?

sp500_data = sp500.history(period="10y")

# Remove unnecessary columns from the DataFrame
sp500_close = sp500_data["Close"]
sp500_data = sp500_data.drop(columns=["Dividends", "Stock Splits", "Close"])



### GRAPH, LEARN TO PRINT ON PAGE ######
plt.plot(sp500_data.index, sp500_close, label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('S&P 500 Closing Prices Over Time')
plt.show()

plt.plot(sp500_data.index, np.log(sp500_data["Volume"]), label='S&P 500 Volume')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('S&P 500 Volume Over Time')
plt.show()

#Taking natural log to reduce memory, still the same "values"
sp500_data["Volume"] = np.log(sp500_data["Volume"])

#Splitting data to create training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sp500_data, sp500_close, test_size=0.25)

#Using regressor instead of Classification to predict numerical values
model = RandomForestRegressor(n_estimators=100, min_samples_split=100, random_state=1, bootstrap=True)
model.fit(x_train,y_train)

preds = model.predict(x_test)


#Created a Dataframe to compare the difference between predicted and actual values
comparison_df = pd.DataFrame({
    "Predicted" : preds,
    "Actual" : y_test,
    "Difference" : preds - y_test
})

#Plotting the difference, the less difference the more accurate
plt.plot(range(len(comparison_df.index)),comparison_df["Difference"])
plt.title("Predicted vs Actual")
plt.show()


#Sci Kit Leanrs perforamnce metrics, r2 and MSE
from sklearn.metrics import mean_squared_error, r2_score

#Error
mse = mean_squared_error(y_test, preds)
print(f'Mean Squared Error: {mse}')

#Precision ?
r2 = r2_score(y_test, preds)
print(f'R-squared: {r2}')

#Claims
#Mean Squared Error: 1893.0502197785963
#R-squared: 0.9985226903190451

#Yesterdays Price 1/19/2024
yesterdays_values = pd.DataFrame({
    "Open" : 4796.28,
    "High" :4842.07,
    "Low" : 4785.87,
    "Volume" : np.log(4287200000)
        },index=[0])

print(model.predict(yesterdays_values))
#Predicted: 4698.58675283
#Actual Price 4839.81
#Difference 141.22324717


from sklearn.tree import export_graphviz
from sklearn import tree


print (tree.export_graphviz(clf.estimators_[0]))

#Copy and paste the code here to view your tree
#https://dreampuf.github.io/GraphvizOnline/

