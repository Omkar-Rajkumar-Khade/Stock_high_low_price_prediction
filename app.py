from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the stock symbol and start date from the form
    stock = request.form['stock']
    start_date = request.form['start_date']
    
    # Download the stock data from Yahoo Finance
    df = yf.download(stock, start=start_date)
    df = df.reset_index()
    if df.index.name == 'Date':
    # Reset the index and move the "Date" column to a regular column
        df = df.reset_index()
        
    print(df)
    
    # Extract day, month, and year values from the Date column
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    
    # Split the data into X (input) and y (output)
    X = df[['Day', 'Month', 'Year', 'Open', 'Close']]
    y = df[['High', 'Low']]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the random forest regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model performance using Root Mean Squared Error (RMSE) and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(r2)
    
    # Get the predicted high and low prices for the next day
    next_day = X.iloc[-1:].copy()
    next_day['Day'] += 1
    next_day_pred = rf_model.predict(next_day)
    next_day_high = next_day_pred[0][0]
    next_day_low = next_day_pred[0][1]
    
    # Render the prediction results template with the predicted high and low prices
    return render_template('result.html', next_day_high=next_day_high, next_day_low=next_day_low, rmse=rmse, r2=r2)

if __name__ == "__main__":
    app.run(debug=True)
