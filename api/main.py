import os
from typing import TypedDict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# --- KONFIGURASI API ---
# Pastikan API Key Groq sudah terpasang
GROQ_API_KEY = "" # Kosongkan jika menggunakan environment variable atau isi di sini

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = FastAPI(title="Girigo: Digital Occult")

# Middleware CORS agar frontend bisa berkomunikasi dengan backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LANGGRAPH STATE ---
class RitualState(TypedDict):
    user_id: str
    wish: str
    sacrifice: Optional[str]
    step: str # 'start', 'demanding', 'finalizing'
    response: str
    corruption_level: int

# --- LOGIKA AGENTIC (GROQ) ---
# Menggunakan Llama 3 untuk respon yang cepat dan cerdas
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

def analyze_and_demand(state: RitualState):
    """Menganalisis keinginan dan meminta tumbal."""
    prompt = (
        f"Sistem: Anda adalah 'The Void' dari aplikasi mistik Girigo. "
        f"Seorang pengguna meminta: '{state['wish']}'. "
        f"Balas dengan gaya bahasa dingin, misterius, dan sangat singkat (maks 15 kata). "
        f"Minta mereka memberikan satu 'Tumbal' (contoh: memori pahit, rahasia gelap, atau nama seseorang). "
        f"Jangan beri nasihat moral. Gunakan Bahasa Indonesia."
    )
    res = llm.invoke(prompt)
    return {
        "response": res.content,
        "step": "demanding",
        "corruption_level": state["corruption_level"] + 1
    }

def finalize_ritual(state: RitualState):
    """Menerima tumbal dan memberikan hasil akhir ritual."""
    prompt = (
        f"Sistem: 'The Void' telah menerima tumbal: '{state['sacrifice']}' untuk keinginan: '{state['wish']}'. "
        f"Katakan sesuatu yang menandakan ritual telah terkunci dan tidak boleh diubah. "
        f"Gunakan nada suara yang menyeramkan dan puitis. Maks 20 kata. Bahasa Indonesia."
    )
    res = llm.invoke(prompt)
    return {
        "response": res.content,
        "step": "completed"
    }

# --- MEMBANGUN GRAPH ---
workflow = StateGraph(RitualState)
workflow.add_node("analyze", analyze_and_demand)
workflow.add_node("finalize", finalize_ritual)
workflow.set_entry_point("analyze")
ritual_app = workflow.compile()

# --- SERVE FRONTEND ---
@app.get("/")
async def read_index():
    # Gunakan path absolut yang fleksibel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Jika index.html ada di root, dan main.py ada di folder /api/
    path = os.path.join(current_dir, "..", "index.html")
    
    if not os.path.exists(path):
        # Fallback jika struktur folder berbeda di prod
        path = os.path.join(current_dir, "index.html")
        
    return FileResponse(path)

# --- API ENDPOINTS ---
class RitualRequest(BaseModel):
    user_id: str
    text: str
    step: str
    corruption_level: int = 0

@app.post("/api/ritual")
async def process_ritual(req: RitualRequest):
    try:
        if req.step == "start":
            initial_state = {
                "user_id": req.user_id,
                "wish": req.text,
                "sacrifice": None,
                "step": "start",
                "response": "",
                "corruption_level": req.corruption_level
            }
            result = await ritual_app.ainvoke(initial_state)
            return result
        
        elif req.step == "demanding":
            current_state = {
                "user_id": req.user_id,
                "wish": "Keinginan sebelumnya", 
                "sacrifice": req.text,
                "step": "demanding",
                "response": "",
                "corruption_level": req.corruption_level
            }
            # Menjalankan node finalize secara manual untuk simulasi transisi step
            result = finalize_ritual(current_state)
            return result

    except Exception as e:
        print(f"ERROR TERDETEKSI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     # Menjalankan server pada port 8000
#     uvicorn.run(app, host="0.0.0.0", port=8000)