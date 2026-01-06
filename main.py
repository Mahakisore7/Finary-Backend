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
from pydantic import BaseModel

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

PRIMARY_CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health"]
FALLBACK_CATEGORY = "Misc"

# CRITICAL: Dual-naming convention to fix 404 errors
GEMINI_MODEL_LANGCHAIN = "models/gemini-2.5-flash" 
GEMINI_MODEL_SDK = "gemini-2.5-flash"

class ChatRequest(BaseModel):
    message: str

# --- 2. SERVICE INITIALIZATIONS ---
app.add_middleware(
    CORSMiddleware,
    # Standard origins for local and production deployment
    allow_origins=["https://finary-ten.vercel.app", "http://localhost:3000", "http://127.0.0.1:3000"], 
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
    # Using 'models/' prefix for LangChain compatibility
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_LANGCHAIN, google_api_key=AI_KEY, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, 
        agent_type="tool-calling", allow_dangerous_requests=True
    )
    print(f"✅ SQL Agent Ready using {GEMINI_MODEL_LANGCHAIN}")
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

# --- 4. AI SCAN ROUTE ---
@app.post("/api/v1/scan-receipt")
async def scan_receipt(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        image_data = await file.read()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        Analyze this receipt. 
        Categorize the expense into one of these: {', '.join(PRIMARY_CATEGORIES)}. 
        If it doesn't fit, use '{FALLBACK_CATEGORY}'.
        Current Date: {current_date}
        Return ONLY JSON: {{"amount": number, "category": string, "description": string, "date": "YYYY-MM-DD"}}
        """
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_SDK,
            contents=[
                types.Part.from_text(text=prompt), 
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        
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

# --- 5. AI VOICE ROUTE ---
@app.post("/api/v1/voice-entry")
async def voice_entry(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        audio_data = await file.read()
        current_date = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""
        Extract transaction from audio. Categorize into: {', '.join(PRIMARY_CATEGORIES)}. 
        Return ONLY JSON: {{"amount": number, "category": string, "description": string, "date": "YYYY-MM-DD", "transcript": string}}
        """
        
        ai_response = client.models.generate_content(
            model=GEMINI_MODEL_SDK,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        data = json.loads(ai_response.text)
        final_category = data.get('category') if data.get('category') in PRIMARY_CATEGORIES else FALLBACK_CATEGORY

        db_entry = {
            "user_id": user.id, 
            "amount": float(data.get('amount', 0)), 
            "category": final_category, 
            "description": f"Voice: {data.get('description', 'Expense')}", 
            "transaction_date": data.get('date', current_date)
        }
        
        supabase.table("transactions").insert(db_entry).execute()
        return {"status": "success", "transcript": data.get('transcript', ''), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. AI CHAT ROUTE (SECURED) ---
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest, x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="User identity missing")

    if not agent_executor:
        return {"answer": "AI Agent offline. Check database connection."}

    # CRITICAL: Multi-user data isolation via prompt injection
    privacy_prompt = f"""
    You are a financial assistant for user_id: {x_user_id}.
    CRITICAL: Every SQL query you write MUST include 'WHERE user_id = '{x_user_id}'' 
    to ensure you ONLY access data belonging to this specific user. 
    Never return totals or descriptions belonging to any other user_id.
    
    User Question: {request.message}
    """
    
    try:
        # Execute the agent chain
        response = agent_executor.invoke({"input": privacy_prompt})
        
        # --- ROBUST OUTPUT PARSING ---
        # The agent returns a dict with an 'output' key
        raw_output = response.get("output", "I couldn't process that query.")

        # If the output is a list of parts (like in your screenshot), extract the text
        if isinstance(raw_output, list):
            # Try to get the 'text' field from the first item, otherwise stringify the whole thing
            final_answer = raw_output[0].get('text', str(raw_output)) if raw_output else ""
        elif isinstance(raw_output, dict):
            final_answer = raw_output.get('text', str(raw_output))
        else:
            final_answer = str(raw_output)

        # Return only the clean string to the frontend
        return {"answer": final_answer}

    except Exception as e:
        print(f"Agent Error: {e}")
        return {"answer": "I'm having trouble connecting. Ensure the backend is live and database is connected."}

# --- 7. PROACTIVE INSIGHTS ---
@app.get("/api/v1/proactive-insight")
async def get_insight(user=Depends(get_current_user)):
    try:
        # Strictly query only current user's data
        res = supabase.table("transactions").select("*").eq("user_id", user.id).order("transaction_date", desc=True).limit(10).execute()
        if not res.data:
            return {"insight": "Start adding expenses to unlock personalized AI insights!"}

        prompt = f"Analyze these transactions and give a 1-sentence tip: {json.dumps(res.data)}"
        response = client.models.generate_content(model=GEMINI_MODEL_SDK, contents=[types.Part.from_text(text=prompt)])
        return {"insight": response.text}
    except Exception as e:
        print(f"Insight Error: {e}")
        return {"insight": "Keep tracking your spending to see smart trends."}

@app.get("/health")
def health(): return {"status": "Online"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)