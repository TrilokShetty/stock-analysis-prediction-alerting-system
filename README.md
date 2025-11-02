

# Stock Analysis & Prediction Dashboard

This project is a multi-page Streamlit web application for stock analysis, machine learning-based forecasting, and automated price alerts.

### Working Link
https://stock-analysis-prediction-alerting-system.streamlit.app

### Demo
* **Video**  

https://github.com/user-attachments/assets/b202044e-c2b9-4705-8c78-0942d7f21490  

* **Forecasting**

<img width="1452" height="796" alt="image" src="https://github.com/user-attachments/assets/b1353a6e-8121-4b49-bf5f-2e22908e9234" />

* **Alert Message**
<img width="678" height="448" alt="image" src="https://github.com/user-attachments/assets/375aca99-01e3-4b65-8eb4-b2e361687d92" />







-----

## Features

  * **Fundamental Analysis:** View key financial data of a selected stock.
  * **Price Forecasting:** Uses an **LSTM** model to predict the next day's closing price.
  * **Automated Alerts:** Triggers notifications based on custom price targets using a serverless backend.

-----

## Technical Overview

### 1\. The Dashboard (Streamlit)

  * `Home.py`: The main entry point for the application.
  * `pages/`: Contains the individual app pages for each feature.

### 2\. ML Model Selection (Jupyter Notebook)

  * `Stock-Prediction.ipynb`: Contains the analysis and comparison of various models (SVM, Random Forest, etc.). **LSTM** was selected for its superior performance on this time-series data.

### 3\. Serverless Alerting System

  * **AWS Lambda** and **AWS EventBridge** are used to run scheduled, serverless checks against user-defined price alerts.

### Key Technologies

  * **Frontend:** Streamlit, Plotly
  * **Backend & Data:** Python, Pandas
  * **Machine Learning:** Scikit-learn, TensorFlow/Keras
  * **Database:** PostgreSQL
  * **Cloud/Infra:** AWS Lambda, AWS EventBridge

-----

## How to Run Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/TrilokShetty/stock-analysis-prediction-alerting-system.git
    cd stock-analysis-prediction-alerting-system
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**

      * Set up your PostgreSQL database credentials. ( use Neon or Supabase DBaaS to use alerting system)
      * Configure AWS credentials for the alert system (Lambda, EventBridge).

5.  **Run the app:**

    ```bash
    streamlit run Home.py
    ```
