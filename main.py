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
GEMINI_MODEL = "gemini-1.5-flash" 

class ChatRequest(BaseModel):
    message: str

# --- 2. SERVICE INITIALIZATIONS ---
app.add_middleware(
    CORSMiddleware,
    # Added 127.0.0.1 and production URLs for full compatibility
    allow_origins=["https://finary-ten.vercel.app", "http://localhost:3000", "http://127.0.0.1:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(SB_URL, SB_KEY)
client = genai.Client(api_key=AI_KEY)

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
        # Decodes the JWT from Supabase to get user details
        user_response = supabase.auth.get_user(token)
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

# --- 4. AI ROUTES ---
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest, x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="User identity missing")

    # This prompt forces the AI to ONLY query this specific user's data
    privacy_prompt = f"""
    You are a financial assistant for user_id: {x_user_id}.
    CRITICAL: Every SQL query you write MUST include 'WHERE user_id = '{x_user_id}'' 
    to ensure you ONLY access THIS user's data. 
    Never return totals or descriptions belonging to any other user.
    
    User Question: {request.message}
    """
    try:
        response = agent_executor.invoke({"input": privacy_prompt})
        return {"answer": response["output"]}
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"answer": "I couldn't process that query. Please try again."}

@app.get("/api/v1/proactive-insight")
async def get_insight(user=Depends(get_current_user)):
    try:
        # Strictly query only this specific user's data
        res = supabase.table("transactions").select("*").eq("user_id", user.id).order("transaction_date", desc=True).limit(10).execute()
        if not res.data:
            return {"insight": "Start adding expenses to unlock personalized AI insights!"}

        prompt = f"Analyze these transactions and give a 1-sentence tip: {json.dumps(res.data)}"
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[types.Part.from_text(text=prompt)])
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