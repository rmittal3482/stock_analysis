# stock_analysis
This project presents a web-based stock analysis and prediction application using Python, Flask, and machine learning. Users can enter a stock ticker and date range to visualize stock trends, calculate technical indicators (Moving Averages, RSI, MACD), and predict the next day's closing price using linear regression. Additionally, the application integrates real-time financial news related to the queried stock, enhancing context-aware analysis.

#Technologies Used

Frontend: HTML, Bootstrap, Bokeh (for interactive plots)

Backend: Python, Flask

Libraries: yfinance, pandas, scikit-learn, requests

Visualization: Bokeh

Machine Learning: Linear Regression (scikit-learn)

#Key Features

Stock Price Visualization

Line charts for closing prices, moving averages, Bollinger Bands, RSI, MACD.

Prediction

Linear Regression predicts the next day’s closing price based on past trends.

Downloadable Data

Users can download stock data in CSV or Excel format.

News Feed Integration

Financial news headlines are fetched using the NewsAPI related to the selected stock.

