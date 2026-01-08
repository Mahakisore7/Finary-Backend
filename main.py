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

# Updated to stable 2.0 series for 2026 production
GEMINI_MODEL_LANGCHAIN = "models/gemini-3-flash" 
GEMINI_MODEL_SDK = "gemini-3-flash"

class ChatRequest(BaseModel):
    message: str

# --- 2. SERVICE INITIALIZATIONS ---
app.add_middleware(
    CORSMiddleware,
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
        # Dynamic date injection
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")
        
        prompt = f"""
        Analyze this receipt image. 
        Categorize into: {', '.join(PRIMARY_CATEGORIES)}. Use '{FALLBACK_CATEGORY}' if unsure.
        
        TEMPORAL CONTEXT:
        - The date today is {current_date_str}.
        - Extract the date from the receipt. 
        - If the receipt date is missing or unreadable, default to {current_date_str}.
        
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
            "transaction_date": data.get('date', current_date_str)
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
        # Dynamic date injection
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")

        prompt = f"""
        Extract transaction from audio.
        TEMPORAL REASONING:
        - Today is {current_date_str} (Day: {now.strftime('%A')}).
        - If the user says "today", use {current_date_str}.
        - If they say "yesterday", calculate the date relative to {current_date_str}.
        - If no date is mentioned, default to {current_date_str}.
        
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
            "transaction_date": data.get('date', current_date_str)
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

    # Dynamic date injection for relative queries (e.g., "this week")
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_day = now.strftime("%A")

    privacy_prompt = f"""
    You are a friendly financial assistant for user_id: {x_user_id}.
    
    TEMPORAL CONTEXT:
    - Today is {current_day}, {current_date_str}.
    - When the user asks about "today", "this week", or "this month", use this date as your reference.
    
    SECURITY RULE:
    - Every SQL query you write MUST include 'WHERE user_id = '{x_user_id}'' 
    - Never return totals or data belonging to other user IDs.
    
    OUTPUT FORMATTING:
    1. NEVER mention the string '{x_user_id}' or 'user ID' in the response.
    2. Provide a clean, human-readable summary.
    
    User Question: {request.message}
    """
    
    try:
        response = agent_executor.invoke({"input": privacy_prompt})
        raw_output = response.get("output", "I couldn't process that query.")

        if isinstance(raw_output, list):
            final_answer = raw_output[0].get('text', str(raw_output)) if raw_output else ""
        elif isinstance(raw_output, dict):
            final_answer = raw_output.get('text', str(raw_output))
        else:
            final_answer = str(raw_output)

        return {"answer": final_answer}
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"answer": "I'm having trouble analyzing your data. Please try again."}

# --- 7. PROACTIVE INSIGHTS ---
@app.get("/api/v1/proactive-insight")
async def get_insight(user=Depends(get_current_user)):
    try:
        # 1. Fetch recent transactions
        res = supabase.table("transactions") \
            .select("*") \
            .eq("user_id", user.id) \
            .order("transaction_date", desc=True) \
            .limit(15) \
            .execute()
        
        transactions = res.data
        if not transactions:
            return {"insight": "Start logging your 2026 expenses to see AI-driven trends!"}

        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")

        # 2. STRICT PROMPT: Forbids internal reasoning in the output
        prompt = f"""
        You are a pro financial advisor. Analyze these transactions: {json.dumps(transactions)}.
        
        STRICT RULES:
        1. OUTPUT ONLY THE TIP: Do not show your analysis, data cleaning steps, or timeline logic.
        2. NO PREAMBLE: Do not start with "Here is an analysis..." or "Based on your data...".
        3. DATE CONTEXT: Today is {current_date_str}. Use this to find the most recent trends.
        4. CATEGORY NORMALIZATION: Treat 'food' and 'Food' as identical behind the scenes.
        
        FINAL TASK: Provide exactly ONE actionable, encouraging financial tip. 
        Example: "Your food spending is up this week; consider meal prepping to save 20%."
        
        Constraint: Max 20 words.
        """
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_SDK, 
            contents=[types.Part.from_text(text=prompt)]
        )
        
        # Clean up any accidental leading/trailing quotes or whitespace
        return {"insight": response.text.strip().replace('"', '')}
        
    except Exception as e:
        print(f"Insight Error: {e}")
        return {"insight": "Track your daily coffee spending to find hidden savings!"}

@app.get("/health")
def health(): return {"status": "Online"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)