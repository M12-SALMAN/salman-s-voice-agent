import pandas as pd
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import base64
import asyncio
import edge_tts

# --- Mic Library ---
from streamlit_mic_recorder import speech_to_text

# Telemetry disable karne ke liye
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ==================================================================
# 1. PAGE CONFIGURATION & PROFESSIONAL THEME (Completely Changed UI)
# ==================================================================
st.set_page_config(page_title="Customer Assistant Dashboard", layout="wide", page_icon="🎧")

# --- PROFESSIONAL CORPORATE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Professional Dark Blue Buttons */
    .stButton>button {
        background: #0f172a !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.2);
        transition: all 0.3s ease;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: #1e293b !important;
        transform: translateY(-2px);
    }
    
    /* Quick Suggestion Outline Buttons */
    div[data-testid="stHorizontalBlock"] button {
        background: white !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #f8fafc !important;
        border-color: #0f172a !important;
    }

    /* Clean Chat Message Bubbles */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
    }
    
    /* Assistant vs User Marking */
    div[data-testid="chat-message-assistant"] {
        border-left: 4px solid #3b82f6; /* Corporate Blue */
    }
    div[data-testid="chat-message-user"] {
        background-color: #f8fafc;
        border-right: 4px solid #64748b;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: transparent; border-radius: 6px;
        padding: 10px 25px; font-weight: 600; color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: white !important; border: none;
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

DATA_FOLDER = "./data"
DOCS_FOLDER = "./documents"
DB_PATH = "./inventory.db"
LOG_DB_PATH = "./system_logs.db"

os.makedirs(DOCS_FOLDER, exist_ok=True)

# ==================================================================
# 2. MALE VOICE GENERATION (URDU - LARKAY KI AWAAZ)
# ==================================================================
def play_voice_male(text):
    async def _generate():
        # ur-PK-AsadNeural (Male Urdu Voice)
        communicate = edge_tts.Communicate(text, "ur-PK-AsadNeural")
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
# 3. LOGGING DATABASE
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
    return 1 if result is None else result + 1

def auto_update_summary(session_id, user_queries_list):
    if not user_queries_list: return
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
    except: pass

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
# 4. EXCEL -> SQLite (Dynamic Table Loading)
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
SCHEMA_TEXT = "\n".join(f"- Table '{t}': columns = {', '.join(c)}" for t, c in SCHEMA.items())

# --- TOOLS ---
@tool
def search_item_fuzzy(table_name: str, search_query: str) -> str:
    """Kisi product, item, ya accessory ki detail ke liye."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except: return f"Table nahi mili. Available tables hain: {list(SCHEMA.keys())}"
    finally: conn.close()
        
    search_words = search_query.lower().replace("-", " ").split()
    results = [row for row in rows if all(word in " ".join(str(v).lower() for v in row) for word in search_words)]
            
    if not results: return f"Koi record nahi mila."
        
    lines = [" | ".join(columns)]
    for row in results[:10]: lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

@tool
def run_sql_query(query: str) -> str:
    """Complex analysis (COUNT, SUM) nikalne ke liye."""
    query_clean = query.strip()
    if not query_clean.lower().startswith("select"): return "Sirf SELECT queries allowed hain."
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(query_clean)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except Exception as e: return f"SQL error: {e}"
    finally: conn.close()

    if not rows: return "Query se koi result nahi mila."
    lines = [" | ".join(columns)]
    for row in rows[:10]: lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

tools = [search_item_fuzzy, run_sql_query]

# ==================================================================
# 5. PROMPT + AGENT (Updated to Customer Assistant, No Name)
# ==================================================================
system_prompt = f"""Tum ek professional 'Customer Assistant' ho. Tumhara koi personal naam nahi hai.

Database mein yeh tables maujood hain:
{SCHEMA_TEXT}

Rule 0: HAMESHA pehle tool call karo.
Rule 1: Jawab Roman Urdu mein do. Friendly aur short rakho.
Rule 2: Tumhara introduction sirf itna hona chahiye: "Assalam o Alaikum! Main aap ki kya madad kar sakta hoon?" Apna naam nahi batana.
Rule 3: Akhir mein poocho "Kya aap ko aur koi maloomat chahiye?"
Rule 4: STRICT RESTRICTION: Apni general knowledge se jawab bilkul nahi dena. Data na mile to mazzrat kar lo.
"""

@st.cache_resource
def get_agent():
    return create_react_agent(model=llm, tools=tools)

agent = get_agent()

# ==================================================================
# 6. STREAMLIT WEB UI & ADMIN LOGIC (Professional Layout)
# ==================================================================
init_logging_db()

# --- PROFESSIONAL SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎧 Customer Assistant")
    st.caption("AI-Powered Support System")
    st.divider()
    
    st.markdown("📍 **Head Office:** Main Branch")
    st.markdown("📞 **Support Line:** 111-XXX-XXX")
    st.markdown("⏰ **Active Hours:** 9:00 AM - 6:00 PM")
    st.divider()
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in ["session_id", "chat_history", "display_msgs", "user_queries"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
        
    st.divider()
    with st.expander("🔒 System Admin Login"):
        admin_password = st.secrets.get("ADMIN_PASSWORD", "")
        user_pass = st.text_input("Enter Passkey", type="password")

is_admin = (user_pass == admin_password and admin_password != "")

# Top Header
col1, col2 = st.columns([0.8, 0.2])
with col1: st.title("Customer Support Portal")
with col2:
    if is_admin: st.success("🟢 Admin Online")

# Session State Initialize (No Name Mentioned in Greeting)
if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()

if "chat_history" not in st.session_state:
    welcome_message = "Assalam o Alaikum! Main aap ki kya madad kar sakta hoon?"
    st.session_state.chat_history = [SystemMessage(content=system_prompt), AIMessage(content=welcome_message)]
    st.session_state.display_msgs = [{"role": "assistant", "content": welcome_message, "audio": None}]
    st.session_state.user_queries = []

if is_admin:
    tab_chat, tab_db, tab_logs = st.tabs(["💬 Assistant Chat", "📊 Database Records", "📝 Session Logs"])
else:
    tab_chat = st.container()

# --- TAB 1: Chat Interface ---
with tab_chat:
    
    # Display previous messages
    for msg in st.session_state.display_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

    quick_query = None

    # --- Quick Suggestions (Generic/Professional) ---
    if len(st.session_state.user_queries) == 0:
        st.write("💡 **Frequently Asked Questions:**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("Product ki price bata dein?", use_container_width=True): quick_query = "Product ki price bata dein?"
        with sc2:
            if st.button("Available colors aur details?", use_container_width=True): quick_query = "Available colors aur details?"
        with sc3:
            if st.button("Installment plan ki maloomat?", use_container_width=True): quick_query = "Installment plan ki maloomat?"

    # --- 🎙️ Mic aur Text Input ---
    col_mic, col_txt = st.columns([0.10, 0.90])
    with col_mic:
        spoken_text = speech_to_text(language='ur-PK', start_prompt="🎙️", stop_prompt="🔴", just_once=True, key='STT')
    with col_txt:
        written_text = st.chat_input("Apna masla ya sawal yahan type karein...")
    
    question = quick_query or written_text or spoken_text

    if question:
        st.chat_message("user").markdown(question)
        st.session_state.display_msgs.append({"role": "user", "content": question, "audio": None})
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.user_queries.append(question)

        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[-8:]

        with st.spinner("Data check kiya ja raha hai..."):
            result = agent.invoke({"messages": st.session_state.chat_history})
            final_message = result["messages"][-1].content
            
            # Male Voice Generate Karna (Asad)
            audio_bytes = play_voice_male(final_message)
            
            with st.chat_message("assistant"):
                st.markdown(final_message)
                if audio_bytes:
                    autoplay_audio(audio_bytes) 
                    st.audio(audio_bytes, format="audio/mp3") 
            
            st.session_state.display_msgs.append({"role": "assistant", "content": final_message, "audio": audio_bytes})
            st.session_state.chat_history = list(result["messages"])
            auto_update_summary(st.session_state.session_id, st.session_state.user_queries)
        
        st.rerun()

# --- TAB 2 & 3: Admin Tabs ---
if is_admin:
    with tab_db:
        st.subheader("📊 System Database")
        conn = sqlite3.connect(DB_PATH)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric(label="Total Data Tables", value=len(SCHEMA.keys()))
        with m2: st.metric(label="Agent Status", value="Online")
        with m3: st.metric(label="Total Support Sessions", value=st.session_state.session_id)
            
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        if tables:
            selected_table = st.selectbox("📌 Select Table to View:", tables)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
        conn.close()
        
    with tab_logs:
        st.subheader("📝 Live Session Logs")
        log_conn = sqlite3.connect(LOG_DB_PATH)
        try:
            chat_df = pd.read_sql_query("SELECT session_id, timestamp, summary as user_questions FROM chat_summaries ORDER BY session_id DESC", log_conn)
            st.dataframe(chat_df, use_container_width=True, hide_index=True)
        except: st.info("Abhi tak koi log entry nahi hai.")
        finally: log_conn.close()
