from flask import Flask, render_template, request, send_file, redirect, url_for
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.resources import INLINE
from bokeh.layouts import column
import requests
import io

app = Flask(__name__)
NEWS_API_KEY = 'b231e95a46544c2fa517e71974e3f0a7'

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short=12, long=26, signal=9):
    ema_short = data.ewm(span=short, adjust=False).mean()
    ema_long = data.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def get_news(ticker):
    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        articles = response.json().get('articles', [])
        return articles[:5]  # Top 5 news articles
    except Exception as e:
        print("Error fetching news:", e)
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    ticker = ''
    error = None
    script = div = ""
    start = ''
    end = ''
    news = []

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        start = request.form['start']
        end = request.form['end']

        df = yf.download(ticker, start=start, end=end)
        if df.empty or df.isnull().all().all():
            error = "No data found for the selected ticker and date range. Please try again."
            return render_template('index.html', error=error)

        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['MB20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['MB20'] + (2 * df['STD20'])
        df['LowerBand'] = df['MB20'] - (2 * df['STD20'])
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])

        df['Prediction'] = df[['Close']].shift(-1)
        X = df[['Close']][:-1]
        y = df['Prediction'][:-1]
        if len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if len(X_train) > 1 and len(X_test) > 0:
                model = LinearRegression()
                model.fit(X_train, y_train)
                prediction = round(model.predict(df[['Close']].tail(1))[0], 2)

        df['Daily Return'] = df['Close'].pct_change() * 100
        df['Cumulative Return'] = (1 + df['Close'].pct_change()).cumprod() - 1
        df['Cumulative Return'] *= 100
        df = df.dropna().copy()
        df.reset_index(inplace=True)

        global latest_df
        latest_df = df

        news = get_news(ticker)

        source = ColumnDataSource(data=dict(
            date=df['Date'],
            close=df['Close'],
            ma7=df['MA7'],
            ma30=df['MA30'],
            volume=df['Volume'],
            daily_return=df['Daily Return'],
            cumulative_return=df['Cumulative Return'],
            mb20=df['MB20'],
            upper_band=df['UpperBand'],
            lower_band=df['LowerBand'],
            rsi=df['RSI'],
            macd=df['MACD'],
            signal=df['Signal']
        ))

        p1 = figure(x_axis_type='datetime', width=800, height=350, title=f"{ticker} Price + Indicators",
                    tools="xwheel_zoom,xpan,reset", active_scroll='xwheel_zoom')
        p1.line('date', 'close', source=source, color='navy', legend_label='Close')
        p1.line('date', 'ma7', source=source, color='green', legend_label='MA7')
        p1.line('date', 'ma30', source=source, color='orange', legend_label='MA30')
        p1.line('date', 'mb20', source=source, color='black', legend_label='SMA20', line_dash='dashed')
        p1.line('date', 'upper_band', source=source, color='red', legend_label='Upper Band')
        p1.line('date', 'lower_band', source=source, color='red', legend_label='Lower Band')
        p1.add_tools(HoverTool(tooltips=[("Date", "@date{%F}"), ("Close", "@close{$0.00}")],
                               formatters={'@date': 'datetime'}, mode='vline'))
        p1.legend.location = "top_left"

        p2 = figure(x_axis_type='datetime', width=800, height=300, x_range=p1.x_range,
                    title=f"{ticker} Volume")
        p2.vbar(x='date', top='volume', source=source, width=0.9, color='grey')

        p3 = figure(x_axis_type='datetime', width=800, height=300, x_range=p1.x_range,
                    title=f"{ticker} Daily Return (%)")
        p3.line('date', 'daily_return', source=source, color='purple')

        p4 = figure(x_axis_type='datetime', width=800, height=300, x_range=p1.x_range,
                    title=f"{ticker} Cumulative Return (%)")
        p4.line('date', 'cumulative_return', source=source, color='teal')

        p5 = figure(x_axis_type='datetime', width=800, height=250, x_range=p1.x_range, title=f"{ticker} RSI")
        p5.line('date', 'rsi', source=source, color='orange')
        p5.y_range.start = 0
        p5.y_range.end = 100
        p5.renderers.extend([
            Span(location=70, dimension='width', line_color='red', line_dash='dashed', line_width=1),
            Span(location=30, dimension='width', line_color='green', line_dash='dashed', line_width=1)
        ])

        p6 = figure(x_axis_type='datetime', width=800, height=250, x_range=p1.x_range,
                    title=f"{ticker} MACD vs Signal Line")
        p6.line('date', 'macd', source=source, color='blue', legend_label='MACD')
        p6.line('date', 'signal', source=source, color='red', legend_label='Signal')
        p6.legend.location = "top_left"

        layout = column(p1, p2, p3, p4, p5, p6)
        script, div = components(layout)

    return render_template('index.html', prediction=prediction, ticker=ticker, error=error,
                           bokeh_script=script, bokeh_div=div,
                           bokeh_css=INLINE.render_css(), bokeh_js=INLINE.render_js(),
                           start=start, end=end, show_downloads=True, news=news)

@app.route('/download/csv')
def download_csv():
    global latest_df
    if not latest_df.empty:
        csv_buffer = io.StringIO()
        latest_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='stock_data.csv'
        )
    return redirect(url_for('index'))

@app.route('/download/xlsx')
def download_xlsx():
    global latest_df
    if not latest_df.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            latest_df.to_excel(writer, index=False, sheet_name='Stock Data')
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='stock_data.xlsx'
        )
    return redirect(url_for('index'))

if __name__ == '__main__':
    latest_df = pd.DataFrame()
    app.run(debug=True)
