import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def audit_transactions(csv_filepath):
    """
    Reads bank statement CSV and identifies:
    1. Isolation Forest Outliers
    2. Duplicate UPI Debits
    3. Merchant Price Creep (Silent Hikes)
    """
    df = pd.read_csv(csv_filepath)
    
    # Clean Debit column values
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0.0)
    
    # -------------------------------------------------------------------
    # 1. Flag Duplicate UPI Debits (Same description & debit amount)
    # -------------------------------------------------------------------
    duplicates = df[df.duplicated(subset=['Date', 'Description', 'Debit'], keep=False) & (df['Debit'] > 0)]
    
    # -------------------------------------------------------------------
    # 2. Detect Price Creep (Silent Subscription Hikes over time)
    # -------------------------------------------------------------------
    # Calculate difference between maximum price paid and initial price paid per merchant description
    debit_transactions = df[df['Debit'] > 0].copy()
    price_trends = debit_transactions.groupby('Description')['Debit'].agg(['min', 'max', 'count']).reset_index()
    
    # Flag merchants where the price increased by > 10% on recurring payments
    price_hikes = price_trends[(price_trends['count'] >= 2) & (price_trends['max'] > price_trends['min'] * 1.10)]
    
    # -------------------------------------------------------------------
    # 3. ML Anomaly Detection (Isolation Forest)
    # -------------------------------------------------------------------
    features = debit_transactions[['Debit']]
    if len(features) > 0:
        model = IsolationForest(contamination=0.05, random_state=42)
        debit_transactions['is_anomaly'] = model.fit_predict(features)
        ml_anomalies = debit_transactions[debit_transactions['is_anomaly'] == -1]
    else:
        ml_anomalies = pd.DataFrame()
        
    return {
        "duplicates": duplicates,
        "price_hikes": price_hikes,
        "ml_anomalies": ml_anomalies
    }

if __name__ == "__main__":
    results = audit_transactions("data/hdfc_sample_statement.csv")
    
    print("\n--- 🚨 DUPLICATE DEBITS DETECTED ---")
    print(results['duplicates'][['Date', 'Description', 'Debit']])
    
    print("\n--- 📈 SUBSCRIPTION PRICE HIKES DETECTED ---")
    print(results['price_hikes'])