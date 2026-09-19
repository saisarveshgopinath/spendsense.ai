from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# Import your existing backend modules
from ml_model import audit_transactions
from normalize import clean_transaction  # <--- Updated function import name!

app = FastAPI(title="SpendSense AI API")

# Enable CORS for React Frontend (Madu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "SpendSense AI Engine Online"}

@app.post("/api/upload")
async def process_statement(file: UploadFile = File(...)):
    # Read uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Save temporary CSV for processing
    temp_path = "data/uploaded_statement.csv"
    df.to_csv(temp_path, index=False)
    
    # 1. Run Regex Normalization on Description column using normalize.py
    if 'Description' in df.columns:
        df['Clean_Merchant'] = df['Description'].apply(lambda x: clean_transaction(str(x))['merchant'])
        df['Category'] = df['Description'].apply(lambda x: clean_transaction(str(x))['category'])
    
    # 2. Run ML Anomaly Detection Engine
    audit_results = audit_transactions(temp_path)
    
    return {
        "status": "success",
        "duplicate_count": len(audit_results['duplicates']),
        "duplicates": audit_results['duplicates'].to_dict(orient="records"),
        "price_hikes": audit_results['price_hikes'].to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)