from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# Import your modules
from ml_model import audit_transactions
from normalize import clean_upi_description # If Dhurshan created a cleaning function
from rag_chatbot import query_statement       # If Dhurshan created a query function

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
    # Read uploaded CSV / file into memory
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Save temporary file for ML audit
    temp_path = "data/uploaded_statement.csv"
    df.to_csv(temp_path, index=False)
    
    # Run your ML Anomaly Engine
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