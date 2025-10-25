
import os
import psycopg2
import requests
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


#database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.environ['DB_HOST'],
            dbname=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASS'],
            port=os.environ.get('DB_PORT', 5432)
        )
        print("Database connection successful")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None





#email sending function
def send_email(to_email, username, symbol, condition, target_price, current_price):
    try:
        smtp_server = os.environ['SMTP_SERVER']
        smtp_port = int(os.environ['SMTP_PORT'])
        sender_email = os.environ['EMAIL_SENDER']
        sender_password = os.environ['EMAIL_SENDER_PASSWORD']

        subject = f"Stock Alert Triggered: {symbol}!"
        body = f"""
        Hi {username},

        Your stock alert for {symbol} has been triggered.

        Your Condition: {condition} ₹{target_price:.2f}
        Current Price: ₹{current_price:.2f}

        This alert has now been removed from your active alerts.

        Regards,
        Stock Alert System
        """

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message.as_string())
        print(f"Email sent to {to_email} for {symbol}")

    except Exception as e:
        print(f"Error sending email: {e}")







#main lambda function

def lambda_handler(event, context):
    conn = get_db_connection()
    if not conn:
        return {'statusCode': 500, 'body': 'Failed to connect to DB'}

    all_alerts = []

    try:
        # fetching all alerts and user info
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    a.id, a.user_id, a.symbol, a.target_price, a.alert_condition,
                    u.email, u.username
                FROM alerts a
                JOIN users u ON a.user_id = u.id
            """)
            all_alerts = cur.fetchall()

        if not all_alerts:
            print(" No active alerts found.")
            conn.close()
            return {'statusCode': 200, 'body': 'No active alerts.'}

        unique_symbols = list(set(alert[2] for alert in all_alerts))
        print(f"Found {len(all_alerts)} alerts for {len(unique_symbols)} unique symbols.")
        print(f"Fetching prices for: {', '.join(unique_symbols)}")


        current_prices = {}
        API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
        
        #warning : alphavantage api has only 25 free calls /day
        for symbol in unique_symbols:
            symbol_bse = symbol.replace(".NS", ".BSE")
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol_bse,
                    "apikey": API_KEY
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Check for API rate limit note
                if "Note" in data:
                    print(f"API Note for {symbol_bse}: {data['Note']}")
                    print("!!! You may have hit your API rate limit (25/day) !!!")
                    continue 
                # getting price
                price_str = data.get("Global Quote", {}).get("05. price")
                if price_str:
                    # Store the price using the ORIGINAL .NS symbol
                    current_prices[symbol] = float(price_str)
                else:
                    print(f"No 'Global Quote' data found for {symbol_bse}. Response: {data}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch price for {symbol_bse}: {e}")
            except Exception as e:
                print(f"Error processing data for {symbol_bse}: {e}")

        print(f"Fetched prices for {len(current_prices)} symbols.")


        #Check and process triggered alerts
        triggered_alerts_ids = []

        for alert in all_alerts:
            alert_id, user_id, symbol, target_price, condition, email, username = alert
            current_price = current_prices.get(symbol)

            if not current_price:
                continue  # skip if no price available

            target_price = float(target_price)
            triggered = False

            if condition == 'GT' and current_price > target_price:
                triggered = True
                condition_text = "Price Above"
            elif condition == 'LT' and current_price < target_price:
                triggered = True
                condition_text = "Price Below"

            if triggered:
                print(f"TRIGGERED: {symbol} ({current_price}) for {username}")
                send_email(
                    to_email=email,
                    username=username,
                    symbol=symbol,
                    condition=condition_text,
                    target_price=target_price,
                    current_price=current_price
                )
                triggered_alerts_ids.append(alert_id)

        # delete triggered alerts
        if triggered_alerts_ids:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM alerts WHERE id = ANY(%s)",
                    (triggered_alerts_ids,)
                )
                conn.commit()
            print(f"Removed {len(triggered_alerts_ids)} triggered alerts.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        conn.rollback()
    finally:
        conn.close()

    return {
        'statusCode': 200,
        'body': f'{len(all_alerts)} alerts.'
    }