# 🤖 Finary AI - Core Engine
**FastAPI + LangChain SQL Agent**

This is the AI engine powering Finary. It manages the communication between the Gemini LLM, the PostgreSQL database, and the frontend clients.

## 🧠 AI Integration
* **LLM**: Gemini 2.5 Flash Lite.
* **Agent Framework**: LangChain SQL Agent with `SQLDatabaseToolkit`.
* **Privacy Logic**: Every AI query is strictly isolated using `x-user-id` headers to ensure multi-tenant data security.

## 🛣️ API Endpoints
* `POST /api/v1/chat`: Secure SQL Agent chat.
* `GET /api/v1/proactive-insight`: Gemini-generated spending analysis.
* `POST /api/v1/scan-receipt`: Vision-based receipt extraction.
* `POST /api/v1/voice-entry`: Audio-to-transaction processing.

## 🛠️ Tech Stack
* **Framework**: FastAPI (Python 3.13)
* **AI**: Google GenAI SDK + LangChain
* **ORM**: SQLAlchemy + Psycopg2
* **Hosting**: Render

## ⚙️ Environment Variables
Create a `.env` file:
```env
SUPABASE_URL=your_url
SUPABASE_SERVICE_ROLE_KEY=your_key
GEMINI_API_KEY=your_google_ai_studio_key
DATABASE_URL=postgresql://postgres:[password]@[aws-1-ap-south-1.pooler.supabase.com:6543/postgres](https://aws-1-ap-south-1.pooler.supabase.com:6543/postgres)

# To run 
pip install -r requirements.txt
python main.py
