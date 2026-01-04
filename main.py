import os
import json
import io
import tempfile
import whisper
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

# --- 2. SERVICE INITIALIZATIONS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(SB_URL, SB_KEY)
client = genai.Client(api_key=AI_KEY)

# SQL Agent Initialization
try:
    print("🔌 Connecting to SQL Database...")
    db = SQLDatabase.from_uri(DB_URL)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=AI_KEY, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, 
        agent_type="tool-calling", allow_dangerous_requests=True
    )
    print("✅ SQL Agent Ready using Gemini 2.5 Flash")
except Exception as e:
    print(f"⚠️ SQL Agent Warning: {e}")
    agent_executor = None

print("🚀 Loading Whisper Voice Model...")
voice_model = whisper.load_model("base")

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

# --- 4. AI VISION ROUTE ---
@app.post("/api/v1/scan-receipt")
async def scan_receipt(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        image_data = await file.read()
        prompt = "Extract details from this receipt image and return ONLY JSON: {amount, category, description, date}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_text(text=prompt), types.Part.from_bytes(data=image_data, mime_type=file.content_type)],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        db_entry = {"user_id": user.id, "amount": float(data.get('amount', 0)), "category": data.get('category', 'Misc'), "description": data.get('description', 'AI Scan'), "transaction_date": data.get('date', '2026-01-01')}
        supabase.table("transactions").insert(db_entry).execute()
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. AI VOICE ROUTE ---
@app.post("/api/v1/voice-entry")
async def voice_entry(file: UploadFile = File(...), user=Depends(get_current_user)):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name
        result = voice_model.transcribe(temp_path)
        prompt = f'Extract transaction details from this text and return ONLY JSON: "{result["text"]}"'
        ai_response = client.models.generate_content(model="gemini-2.5-flash", contents=[types.Part.from_text(text=prompt)], config=types.GenerateContentConfig(response_mime_type="application/json"))
        data = json.loads(ai_response.text)
        db_entry = {"user_id": user.id, "amount": float(data.get('amount', 0)), "category": data.get('category', 'Misc'), "description": f"Voice: {data.get('description', 'Expense')}", "transaction_date": data.get('date', '2026-01-04')}
        supabase.table("transactions").insert(db_entry).execute()
        return {"status": "success", "transcript": result['text'], "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

# --- 6. AI CHAT ROUTE (Fixed for React) ---
@app.post("/api/v1/chat")
async def chat_with_data(payload: dict, user=Depends(get_current_user)):
    try:
        user_query = payload.get("message")
        secure_query = f"As a financial assistant for user_id '{user.id}', only query rows where user_id matches '{user.id}'. Question: {user_query}"
        
        response = agent_executor.invoke({"input": secure_query})
        
        # FIX: Extract only the text string so React doesn't crash
        raw_output = response["output"]
        if isinstance(raw_output, list) and len(raw_output) > 0:
            final_answer = raw_output[0].get("text", str(raw_output))
        else:
            final_answer = str(raw_output)

        return {"status": "success", "answer": final_answer}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI failed to query data.")
    

@app.get("/api/v1/proactive-insight")
async def get_insight(user=Depends(get_current_user)):
    try:
        # 1. Fetch recent transactions
        res = supabase.table("transactions").select("*").eq("user_id", user.id).limit(10).execute()
        transactions = res.data

        if not transactions:
            return {"insight": "Start adding expenses to see AI insights!"}

        # 2. Ask Gemini for a proactive "nudge"
        prompt = f"""
        Analyze these transactions and give a 1-sentence proactive financial tip.
        Be specific, helpful, and slightly "cool" or "fintech-pro". 
        Data: {json.dumps(transactions)}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_text(text=prompt)]
        )
        
        return {"insight": response.text}
    except Exception as e:
        return {"insight": "Keep tracking to unlock smart insights."}

@app.get("/health")
def health(): return {"status": "Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)