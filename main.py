import os
import json
import io
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from google import genai
from google.genai import types

# LangChain & SQL Agent Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

load_dotenv()

app = FastAPI(title="Finary AI - Core Engine")

# --- 1. CONFIGURATION ---
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
AI_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Define the anchor categories for your charts and budgets
PRIMARY_CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health"]
FALLBACK_CATEGORY = "Misc"

# Stable 2026 model
GEMINI_MODEL = "gemini-2.5-flash-lite" 

# --- 2. SERVICE INITIALIZATIONS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finary-ten.vercel.app", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(SB_URL, SB_KEY)
client = genai.Client(api_key=AI_KEY)

# SQL Agent Initialization
try:
    print(f"🔌 Connecting to SQL Database...")
    db = SQLDatabase.from_uri(DB_URL)
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=AI_KEY, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, 
        agent_type="tool-calling", allow_dangerous_requests=True
    )
    print(f"✅ SQL Agent Ready using {GEMINI_MODEL}")
except Exception as e:
    print(f"⚠️ SQL Agent Warning: {e}")
    agent_executor = None

# --- 3. AUTH DEPENDENCY ---
async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    try:
        user_response = supabase.auth.get_user(token)
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

# --- 4. AI VISION ROUTE (FLEXIBLE CATEGORIES) ---
@app.post("/api/v1/scan-receipt")
async def scan_receipt(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        image_data = await file.read()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        Analyze this receipt. 
        Categorize the expense into one of these: {', '.join(PRIMARY_CATEGORIES)}. 
        If the expense does not clearly fit into these, use '{FALLBACK_CATEGORY}'.
        Note: Items like snacks, drinks, or groceries must be categorized as 'Food'.
        Current Date: {current_date}
        Return ONLY JSON: {{"amount": number, "category": string, "description": string, "date": "YYYY-MM-DD"}}
        """
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt), 
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        
        # Ensure the category is valid before DB entry
        final_category = data.get('category') if data.get('category') in PRIMARY_CATEGORIES else FALLBACK_CATEGORY
        
        db_entry = {
            "user_id": user.id, 
            "amount": float(data.get('amount', 0)), 
            "category": final_category, 
            "description": data.get('description', 'AI Scan'), 
            "transaction_date": data.get('date', current_date)
        }
        supabase.table("transactions").insert(db_entry).execute()
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. AI VOICE ROUTE (FLEXIBLE CATEGORIES) ---
@app.post("/api/v1/voice-entry")
async def voice_entry(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        audio_data = await file.read()
        current_date = datetime.now().strftime("%Y-%m-%d")
        print(f"🎙️ Audio received at {current_date}")

        prompt = f"""
        Transcribe the audio and extract transaction details. 
        Categorize into: {', '.join(PRIMARY_CATEGORIES)}. 
        If it doesn't fit, use '{FALLBACK_CATEGORY}'.
        Example: 'I spent 50 on snacks' -> Category: 'Food', Description: 'snacks'.
        Return ONLY JSON: {{"amount": number, "category": string, "description": string, "date": "YYYY-MM-DD", "transcript": string}}
        """
        
        ai_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        data = json.loads(ai_response.text)
        print(f"🤖 AI Result: {data}")

        # Code-level category enforcement
        final_category = data.get('category') if data.get('category') in PRIMARY_CATEGORIES else FALLBACK_CATEGORY

        db_entry = {
            "user_id": user.id, 
            "amount": float(data.get('amount', 0)), 
            "category": final_category, 
            "description": f"Voice: {data.get('description', 'Expense')}", 
            "transaction_date": data.get('date', current_date)
        }
        
        result = supabase.table("transactions").insert(db_entry).execute()
        print(f"✅ Supabase Save: {result}")
        
        return {"status": "success", "transcript": data.get('transcript', ''), "data": data}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. AI CHAT ROUTE ---
@app.post("/api/v1/chat")
async def chat_with_data(payload: dict, user=Depends(get_current_user)):
    try:
        user_query = payload.get("message")
        secure_query = f"As a financial assistant for user_id '{user.id}', only query rows where user_id matches '{user.id}'. Question: {user_query}"
        
        response = agent_executor.invoke({"input": secure_query})
        return {"status": "success", "answer": str(response["output"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI failed to query data.")

# --- 7. PROACTIVE INSIGHTS ---
@app.get("/api/v1/proactive-insight")
async def get_insight(user=Depends(get_current_user)):
    try:
        res = supabase.table("transactions").select("*").eq("user_id", user.id).limit(10).execute()
        transactions = res.data

        if not transactions:
            return {"insight": "Start adding expenses to see AI insights!"}

        prompt = f"Analyze these transactions and give a 1-sentence proactive financial tip: {json.dumps(transactions)}"
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[types.Part.from_text(text=prompt)])
        return {"insight": response.text}
    except Exception:
        return {"insight": "Keep tracking to unlock smart insights."}

@app.get("/health")
def health(): return {"status": "Online"}

if __name__ == "__main__":
    import uvicorn
    # Bound to dynamic port for local or future cloud environments
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)