import os

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define standard Indian merchants and messy UPI VPAs
normal_merchants = [
    ("Swiggy", "UPI-SWIGGY-PAYMENT-9821@axisbank"),
    ("Zomato", "UPI-ZOMATO-ORDER-1120@icici"),
    ("Uber India", "UPI-UBER-RIDE-0021@okaxis"),
    ("Amazon India", "UPI-AMAZON-PAY-8831@apl"),
    ("Local Kirana", "UPI-RAMESH-STORES-9842@ybl"),
]

# Set initial parameters
start_date = datetime(2026, 1, 1)
balance = 50000.0
data = []

# 1. Generate normal daily expenses over 60 days
for day in range(60):
    current_date = (start_date + timedelta(days=day)).strftime("%d/%m/%Y")
    
    # Random salary credit on 1st of month
    if day == 0 or day == 31:
        credit = 65000.0
        balance += credit
        data.append([current_date, "ACH C- COMPANY SALARY OCT", 0.0, credit, balance])

    # Add 1-2 daily transactions
    for _ in range(random.randint(1, 2)):
        name, vpa = random.choice(normal_merchants)
        debit = round(random.uniform(150.0, 800.0), 2)
        balance -= debit
        data.append([current_date, vpa, debit, 0.0, balance])

# -------------------------------------------------------------
# 2. INJECT INTENTIONAL ANOMALIES FOR ML & REGEX TESTING
# -------------------------------------------------------------

# Anomaly A: Price Creep (Netflix subscription increases over time)
data.append(["15/01/2026", "UPI-NETFLIX-ENTERTAINMENT@icici", 499.0, 0.0, balance - 499.0])
data.append(["15/02/2026", "UPI-NETFLIX-ENTERTAINMENT@icici", 649.0, 0.0, balance - 1148.0]) # Hikes from ₹499 to ₹649

# Anomaly B: Duplicate UPI Debit Glitch (Two identical debits within seconds)
data.append(["20/02/2026", "UPI-ZOMATO-ORDER-9912@icici", 340.0, 0.0, balance - 1488.0])
data.append(["20/02/2026", "UPI-ZOMATO-ORDER-9912@icici", 340.0, 0.0, balance - 1828.0]) # Duplicate charge!

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Date", "Description", "Debit", "Credit", "Balance"])

# Save files into your data directory
df.to_csv("data/hdfc_sample_statement.csv", index=False)
print("Synthetic Bank Statement created successfully in data/hdfc_sample_statement.csv!")
# Add this line right above df.to_csv:
os.makedirs("data", exist_ok=True)
# Add this line right above df.to_csv:
os.makedirs("data", exist_ok=True)

# Now save your file
df.to_csv("data/hdfc_sample_statement.csv", index=False)
print("Saved successfully into the data folder!")