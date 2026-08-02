import pandas as pd
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
import streamlit as st
import base64
import asyncio
import edge_tts
import io

# --- Mic Library ---
from streamlit_mic_recorder import speech_to_text

# Telemetry disable karne ke liye
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from typing import Any
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ==================================================================
# PAGE CONFIGURATION (Streamlit)
# ==================================================================
st.set_page_config(page_title="AI Customer Assistant", layout="wide", page_icon="⚡")

# ==================================================================
# HIGH-END SaaS FRONTEND DESIGN (Fixed Layout)
# ==================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* ---------------- GLASSMORPHISM AGENT HEADER ---------------- */
    .agent-header-container {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        margin-top: 10px;
    }
    
    .agent-header {
        display: flex;
        align-items: center;
        gap: 20px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 15px 35px;
        border-radius: 60px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    }
    
    .agent-avatar-wrapper {
        position: relative;
    }
    
    .agent-avatar {
        width: 65px;
        height: 65px;
        border-radius: 50%;
        background-color: #f1f5f9;
        object-fit: cover;
        z-index: 2;
        position: relative;
    }
    
    .avatar-ring {
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        z-index: 1;
        animation: spin-pulse 3s linear infinite;
        opacity: 0.7;
    }
    
    @keyframes spin-pulse {
        0% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.08); opacity: 0.9; }
        100% { transform: scale(1); opacity: 0.5; }
    }
    
    .agent-details {
        display: flex;
        flex-direction: column;
    }
    
    .agent-name {
        font-size: 19px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.3px;
    }
    
    .status-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 4px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-text {
        font-size: 12px;
        font-weight: 500;
        color: #64748b;
    }

    /* ---------------- CHAT INTERFACE ---------------- */
    div[data-testid="chat-message-assistant"] {
        background-color: #ffffff !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
        border: 1px solid #f1f5f9 !important;
        color: #334155 !important;
        margin-bottom: 20px;
        max-width: 90%;
    }
    
    div[data-testid="chat-message-user"] {
        background-color: #0f172a !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 15px 20px !important;
        color: #f8fafc !important;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.1) !important;
        margin-bottom: 20px;
        margin-left: auto;
        max-width: 90%;
        border: none !important;
    }
    
    .stChatMessage [data-testid="stIcon"] {
        display: none;
    }

    /* ---------------- AGGRESSIVE INPUT BOX STYLING ---------------- */
    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border-radius: 30px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
        padding: 2px 10px !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(button) {
        display: flex;
        justify-content: center;
        margin-bottom: -15px; 
    }

    .stButton>button {
        background: #0f172a !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(15,23,42,0.15);
        transition: all 0.2s ease;
        font-weight: 600;
        height: 45px;
        padding: 0 25px !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: #1e293b !important;
        box-shadow: 0 6px 20px rgba(15,23,42,0.2);
    }

    audio {
        height: 40px;
        border-radius: 12px;
        outline: none;
        margin-top: 10px;
        width: 100%;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

DATA_FOLDER = "./data"
DOCS_FOLDER = "./documents"
DB_PATH = "./inventory_system.db"
PERSIST_FOLDER = "./vectorstore"
LOG_DB_PATH = "./chat_logs.db"

os.makedirs(DOCS_FOLDER, exist_ok=True)

# ==================================================================
# VOICE GENERATION FUNCTION (Clear & Professional Tone)
# ==================================================================
def play_voice_male(text):
    async def _generate():
        # MadhurNeural ek behtareen, clear aur professional corporate voice hai
        communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="-5%")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
        
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(_generate())
    except Exception as e:
        print(f"Voice generation error: {e}")
        return None

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true" class="stAudio">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# ==================================================================
# LOGGING DATABASE SETUP
# ==================================================================
def init_logging_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_summaries (
            session_id INTEGER PRIMARY KEY,
            timestamp TEXT,
            summary TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_new_session_id() -> int:
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(session_id) FROM chat_summaries')
    result = cursor.fetchone()[0]
    conn.close()
    if result is None:
        return 1
    return result + 1

def auto_update_summary(session_id, user_queries_list):
    if not user_queries_list:
        return
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_text = " | ".join(user_queries_list)
        cursor.execute('''
            INSERT OR REPLACE INTO chat_summaries (session_id, timestamp, summary) 
            VALUES (?, ?, ?)
        ''', (session_id, timestamp, summary_text))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Log Error] Summary auto-update nahi ho saki: {e}")

# ----------------------------------------------------------------
# CACHE MODELS 
# ----------------------------------------------------------------
@st.cache_resource
def load_models():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return llm, embeddings, reranker

llm, embeddings, reranker_model = load_models()

# ==================================================================
# EXCEL -> SQLite 
# ==================================================================
@st.cache_data
def load_excels_to_sqlite() -> dict:
    conn = sqlite3.connect(DB_PATH)
    schema = {}
    if os.path.exists(DATA_FOLDER):
        for file in os.listdir(DATA_FOLDER):
            if file.endswith(".xlsx") and not file.startswith("~$"):
                table_name = file.replace(".xlsx", "").lower().replace(" ", "_")
                df = pd.read_excel(os.path.join(DATA_FOLDER, file))
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                schema[table_name] = df.columns.tolist()
    conn.close()
    return schema

SCHEMA = load_excels_to_sqlite()
SCHEMA_TEXT = "\n".join(
    f"- Table '{table}': columns = {', '.join(cols)}" for table, cols in SCHEMA.items()
)

# --- TOOLS ---
@tool
def search_item_fuzzy(table_name: str, search_query: str) -> str:
    """Kisi product ya accessory ki detail ke liye (Fuzzy Match)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except Exception as e:
        return f"Table nahi mili. Available tables hain: {list(SCHEMA.keys())}"
    finally:
        conn.close()
        
    search_words = search_query.lower().replace("-", " ").split()
    results = []
    for row in rows:
        row_text = " ".join(str(v).lower() for v in row)
        if all(word in row_text for word in search_words):
            results.append(row)
            
    if not results:
        return f"Table '{table_name}' mein '{search_query}' se milta julta koi record nahi mila."
        
    lines = [" | ".join(columns)]
    for row in results[:15]:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

@tool
def run_sql_query(query: str) -> str:
    """Complex analysis (COUNT, SUM) nikalne ke liye."""
    query_clean = query.strip()
    if not query_clean.lower().startswith("select"):
        return "Sirf SELECT queries allowed hain."

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(query_clean)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()

    if not rows:
        return "Query se koi result nahi mila."

    lines = [" | ".join(columns)]
    for row in rows[:20]:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

@tool
def search_documents(query: str) -> str:
    """PDF ya Word files mein se jawab dhoondne ke liye."""
    if 'doc_vectorstore' not in globals() or doc_vectorstore is None:
        return "Abhi koi PDF/Word file 'documents' folder mein maujood nahi hai."

    base_retriever = doc_vectorstore.as_retriever(search_kwargs={"k": 10})
    raw_docs = base_retriever.invoke(query)
    if not raw_docs:
        return "Documents mein koi relevant jawab nahi mila."

    pairs = [(query, doc.page_content) for doc in raw_docs]
    scores = reranker_model.score(pairs)
    doc_score_pairs = list(zip(raw_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_2_docs = [doc for doc, score in doc_score_pairs[:2]]
    return "\n---\n".join(f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in top_2_docs)

tools = [search_item_fuzzy, run_sql_query, search_documents]

# ==================================================================
# PROMPT + AGENT (Human-like Conversational Tone)
# ==================================================================
system_prompt = f"""Tum ek professional Customer Assistant ho. Tumhara naam Salman hai. Tumhara kaam AI call agent ki tarha customers ko smoothly guide karna hai.

Database mein yeh tables maujood hain:
{SCHEMA_TEXT}

Rule 0: HAMESHA pehle tool call karo.
Rule 1: Kisi specific product, part, ya accessory ki detail/price mangi jaye to HAMESHA 'search_item_fuzzy' tool use karo.
Rule 2: Complex analysis (Counting, Totals) ke liye 'run_sql_query' use karo.
Rule 3: Policy, agreement ya document se related sawal ho to 'search_documents' use karo.
Rule 4: Agar sab tools fail ho jayen, to bolo "Maaf kijiye ga, ye maloomat abhi mere paas nahi hai, baraye meharbani support team se rabta kar lein".
Rule 5: SAB SE ZAROORI: Tumhari baat cheet bilkul ek aam insaan (human) ki tarha honi chahiye. AI ya robot ki tarha lamba aur mushkil jawab nahi dena. Chotay, friendly, aur conversational sentences use karo (jaise "Ji bilkul", "Han ji", "Zaroor", "Dekhein").
Rule 6: Jawab Roman Urdu mein do. Price ho to Rs likho.
Rule 7: Pichli baatcheet yaad rakho.
Rule 8: Akhir mein poocho "Kya main aapki mazeed koi help kar sakta hoon?"
Rule 9: STRICT RESTRICTION: Apni general knowledge se jawab bilkul nahi dena. Data na mile to mazzrat kar lo.
"""

@st.cache_resource
def get_agent():
    return create_react_agent(model=llm, tools=tools)

agent = get_agent()

# ==================================================================
# STREAMLIT WEB UI & ADMIN LOGIC
# ==================================================================
init_logging_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]
if "display_msgs" not in st.session_state:
    st.session_state.display_msgs = []
if "user_queries" not in st.session_state:
    st.session_state.user_queries = []

with st.sidebar:
    st.title("🔒 System Settings")
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    user_pass = st.text_input("Enter Passkey", type="password")

is_admin = (user_pass == admin_password and admin_password != "")

if is_admin:
    st.success("🟢 Admin Access Granted")
    tab_chat, tab_db, tab_logs = st.tabs(["💬 Assistant", "📊 Database", "📝 Logs"])
else:
    tab_chat = st.container()

# --- TAB 1: Chat Interface ---
with tab_chat:
    
    left_spacer, center_column, right_spacer = st.columns([1, 2, 1])
    
    with center_column:
        
        st.markdown("""
            <div class="agent-header-container">
                <div class="agent-header">
                    <div class="agent-avatar-wrapper">
                        <div class="avatar-ring"></div>
                        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/People/Man%20Office%20Worker.png" class="agent-avatar" alt="AI Agent">
                    </div>
                    <div class="agent-details">
                        <div class="agent-name">Voice Assistant</div>
                        <div class="status-badge">
                            <div class="status-dot"></div>
                            <div class="status-text">Listening & ready</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.display_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        spoken_text = speech_to_text(
            language='ur-PK', 
            start_prompt="🎙️ Tap to Speak",
            stop_prompt="🔴 Processing...",
            just_once=True,
            key='STT'
        )

        written_text = st.chat_input("Type your message here...")
        
        question = written_text or spoken_text

        if question:
            st.chat_message("user").markdown(question)
            
            st.session_state.display_msgs.append({"role": "user", "content": question})
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.user_queries.append(question)

            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[-8:]

            with st.spinner("AI is generating response..."):
                result = agent.invoke({"messages": st.session_state.chat_history})
                final_message = result["messages"][-1].content
                
                st.chat_message("assistant").markdown(final_message)
                
                audio_data = play_voice_male(final_message)
                if audio_data:
                    autoplay_audio(audio_data)
                    st.audio(audio_data, format="audio/mp3")
                
                st.session_state.display_msgs.append({"role": "assistant", "content": final_message})
                st.session_state.chat_history = list(result["messages"])
                
                auto_update_summary(st.session_state.session_id, st.session_state.user_queries)

# --- TAB 2 & 3: Admin Tabs ---
if is_admin:
    with tab_db:
        st.subheader("📦 Knowledge Base")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            selected_table = st.selectbox("Select Table:", tables)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No databases connected.")
        conn.close()
        
    with tab_logs:
        st.subheader("📝 Conversation Logs")
        log_conn = sqlite3.connect(LOG_DB_PATH)
        try:
            chat_df = pd.read_sql_query("SELECT session_id, timestamp, summary as user_questions FROM chat_summaries ORDER BY session_id DESC", log_conn)
            st.dataframe(chat_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info("No chat history found.")
        finally:
            log_conn.close()
