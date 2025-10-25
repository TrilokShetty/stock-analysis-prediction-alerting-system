# app.py
import streamlit as st
import psycopg2
import pandas as pd
import hashlib

# Database connection function
@st.cache_resource
def get_db_connection():
    
    try:
        conn = psycopg2.connect(st.secrets.postgres.uri)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function to check password
def check_password(hashed_password, user_password):
    return hashed_password == hash_password(user_password)

# Load stock symbols
@st.cache_data
def load_symbols():
    try:
        df = pd.read_csv("symbols.csv")
        df['Display'] = df['Company'] + " (" + df['Symbol'] + ")"
        df['YahooSymbol'] = df['Symbol'] + ".NS"
        return df
    except FileNotFoundError:
        st.error("Symbls.csv not found. Please make sure it's in the same directory.")
        return pd.DataFrame()


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_id = None



symbols_df = load_symbols()

# auth code
if not st.session_state.logged_in:
    st.title("Stock Alert System")
    choice = st.sidebar.selectbox("Login/Signup", ["Login", "Signup"])

    if choice == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")

        if st.sidebar.button("Login"):
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
                    user = cur.fetchone()
                    if user and check_password(user[1], password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = user[0]
                        st.sidebar.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error("Incorrect username or password")
                

    elif choice == "Signup":
        st.sidebar.subheader("Create an Account")
        new_username = st.sidebar.text_input("Username", key="signup_username")
        new_email = st.sidebar.text_input("Email", key="signup_email")
        new_password = st.sidebar.text_input("Password", type="password", key="signup_password")

        if st.sidebar.button("Signup"):
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    try:
                        hashed_pass = hash_password(new_password)
                        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                                    (new_username, new_email, hashed_pass))
                        conn.commit()
                        st.sidebar.success("Account created successfully! Please login.")
                    except psycopg2.IntegrityError:
                        st.sidebar.error("Username or email already exists.")
                    except Exception as e:
                        st.sidebar.error(f"An error occurred: {e}")
                


else:
    st.sidebar.title(f"Welcome, {st.session_state.username} !")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None
        st.rerun()

    st.title("Stock Alert Dashboard")

    
    st.header("Create a New Alert")
    if not symbols_df.empty:
        selected_display = st.selectbox("Select Stock", symbols_df['Display'])
        selected_yahoo_symbol = symbols_df.loc[symbols_df['Display'] == selected_display, 'YahooSymbol'].iloc[0]
        
        # NEW: Alert Condition
        condition = st.radio(
            "Alert me when price is:",
            ("Above", "Below"),
            horizontal=True,
            key="condition_radio"
        )
        
        target_price = st.number_input("Target Price (₹)", min_value=0.01, format="%.2f")

        

        if st.button("Create Alert"):
          alert_condition = "GT" if condition == "Above" else "LT"
          conn = get_db_connection()
          
          if conn:
              with conn.cursor() as cur:
                  try:
                      cur.execute(
                          "SELECT id FROM alerts WHERE user_id = %s AND symbol = %s AND target_price = %s AND alert_condition = %s",
                          (st.session_state.user_id, selected_yahoo_symbol, target_price, alert_condition)
                      )
                      
                      if cur.fetchone():
                          st.warning(f"You already have this exact alert.")
                      else:
                          cur.execute(
                              "INSERT INTO alerts (user_id, symbol, target_price, alert_condition) VALUES (%s, %s, %s, %s)",
                              (st.session_state.user_id, selected_yahoo_symbol, target_price, alert_condition)
                          )
                          conn.commit()
                          st.success(f"Alert created for {selected_yahoo_symbol}!")
                          st.rerun()
                          
                  except Exception as e:
                      st.error(f"Error: {e}")
    else:
        st.warning("Stock symbols could not be loaded.")

    
    st.header("Your Active Alerts")
    
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, symbol, target_price, alert_condition FROM alerts WHERE user_id = %s ORDER BY created_at DESC",
                (st.session_state.user_id,)
            )
            alerts = cur.fetchall()
            
            if not alerts:
                st.info("You have no active alerts.")
            else:
                
                cols = st.columns([3, 3, 1])
                cols[0].markdown("**Symbol**")
                cols[1].markdown("**Condition**")
                cols[2].markdown("**Remove**")
                # st.divider()

                
                for alert in alerts:
                    alert_id, symbol, target_price, condition = alert
                    
                    condition_text = f"ABOVE ₹{target_price}" if condition == 'GT' else f"BELOW ₹{target_price}"
                    
                    cols = st.columns([3, 3, 1])
                    cols[0].text(symbol)
                    cols[1].text(condition_text)
                    
                    # Create a unique key for each button
                    button_key = f"{alert_id}"
                    
                    # Check if this button was pressed
                    if cols[2].button("Remove", key=button_key):
                        # If pressed, delete this alert and rerun
                        with conn.cursor() as del_cur:
                            del_cur.execute("DELETE FROM alerts WHERE id = %s AND user_id = %s",
                                            (alert_id, st.session_state.user_id))
                            conn.commit()
                            st.success(f"Alert for {symbol} removed!")
                            st.rerun()
        