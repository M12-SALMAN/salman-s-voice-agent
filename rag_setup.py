import pandas as pd
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
import streamlit as st
import base64

# --- Voice & Mic Libraries ---
from gtts import gTTS
import io
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
# 1. PAGE CONFIGURATION & VOICE AGENT THEME
# ==================================================================
st.set_page_config(page_title="Shamas Honda - Voice Agent", layout="wide", page_icon="🏍️")

# --- CUSTOM CSS FOR VOICE UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Voice Center UI */
    .voice-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 50px;
        margin-bottom: 50px;
    }
    
    .subtitle-text {
        font-size: 20px;
        color: #555;
        font-style: italic;
        margin-top: 20px;
    }
    
    .salman-reply {
        font-size: 24px;
        font-weight: 600;
        color: #cc0000;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    /* Hide Streamlit default UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

DATA_FOLDER = "./data"
DOCS_FOLDER = "./documents"
DB_PATH = "./shamas_honda.db"
PERSIST_FOLDER = "./vectorstore"
LOG_DB_PATH = "./chat_logs.db"

os.makedirs(DOCS_FOLDER, exist_ok=True)

# ==================================================================
# 2. VOICE GENERATION & FORCE AUTOPLAY FUNCTIONS
# ==================================================================
def play_voice(text):
    try:
        tts = gTTS(text=text, lang='hi')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        print(f"Voice error: {e}")
        return None

def autoplay_audio(audio_bytes):
    """HTML trick to force browser autoplay"""
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# ==================================================================
# 3. LOGGING DATABASE SETUP
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
    except Exception:
        pass

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
# 4. EXCEL -> SQLite 
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
    """Kisi bike, part, ya accessory ki detail ke liye."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except Exception:
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
        return f"Data nahi mila."
        
    lines = [" | ".join(columns)]
    for row in results[:5]:  # Voice agent ke liye short rakha hai
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

@tool
def run_sql_query(query: str) -> str:
    """Complex analysis (COUNT, SUM) nikalne ke liye."""
    query_clean = query.strip()
    if not query_clean.lower().startswith("select"):
        return "Sirf SELECT allowed hai."
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
        return "Data nahi mila."

    lines = [" | ".join(columns)]
    for row in rows[:5]:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)

@tool
def search_documents(query: str) -> str:
    """PDF ya Word files mein se jawab dhoondne ke liye."""
    if 'doc_vectorstore' not in globals() or doc_vectorstore is None:
        return "Documents available nahi hain."
    base_retriever = doc_vectorstore.as_retriever(search_kwargs={"k": 10})
    raw_docs = base_retriever.invoke(query)
    if not raw_docs:
        return "Relevant jawab nahi mila."

    pairs = [(query, doc.page_content) for doc in raw_docs]
    scores = reranker_model.score(pairs)
    doc_score_pairs = list(zip(raw_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_2_docs = [doc for doc, score in doc_score_pairs[:2]]
    return "\n---\n".join(f"{doc.page_content}" for doc in top_2_docs)

tools = [search_item_fuzzy, run_sql_query, search_documents]

# ==================================================================
# 5. PROMPT + AGENT (Voice-Optimized)
# ==================================================================
system_prompt = f"""Tum Shamas Honda, Sialkot ke senior sales dealer ho. Naam: Salman.

Database mein yeh tables maujood hain:
{SCHEMA_TEXT}

Rule 0: HAMESHA pehle tool call karo.
Rule 1: Jawab Roman Urdu mein do.
Rule 2: Tumhara jawab awaaz (voice) mein sunaya jayega, isliye jawab ko mukhtasar (short) aur conversational (bol chal ke andaz mein) rakho.
Rule 3: Lambe tables ya lists mat bolo. Sirf main point aur price batao.
Rule 4: Akhir mein poocho "Aur koi help chahiye?"
Rule 5: Apni general knowledge se jawab bilkul nahi dena. Data na mile to mazzrat kar lo.
"""

@st.cache_resource
def get_agent():
    return create_react_agent(model=llm, tools=tools)

agent = get_agent()

# ==================================================================
# 6. STREAMLIT WEB UI & ADMIN LOGIC
# ==================================================================
init_logging_db()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda_Logo.svg", width=150)
    st.markdown("## Shamas Honda")
    st.caption("AI Voice Agent")
    st.divider()
    if st.button("🗑️ Nayi Baat Shuru Karein", use_container_width=True):
        for key in ["session_id", "chat_history", "user_queries", "latest_response", "latest_user_text"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    with st.expander("🔒 Admin Access"):
        admin_password = st.secrets.get("ADMIN_PASSWORD", "")
        user_pass = st.text_input("Enter Password", type="password")

is_admin = (user_pass == admin_password and admin_password != "")

# Session State Initialize
if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()

if "chat_history" not in st.session_state:
    welcome_message = "Assalam o Alaikum! Main Salman baat kar raha hoon Shamas Honda Sialkot se. Main aapki kya madad kar sakta hoon?"
    st.session_state.chat_history = [
        SystemMessage(content=system_prompt),
        AIMessage(content=welcome_message)
    ]
    st.session_state.latest_response = welcome_message
    st.session_state.latest_user_text = ""
    st.session_state.user_queries = []

if is_admin:
    tab_voice, tab_db, tab_logs = st.tabs(["🎙️ Voice Agent", "📊 Admin Dashboard", "📝 Customer Logs"])
else:
    tab_voice = st.container()

# --- TAB 1: Voice Interface ---
with tab_voice:
    st.markdown("<div class='voice-container'>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda_Logo.svg", width=100)
    st.title("Main Salman Hoon")
    st.caption("Mic button par click karein aur apna sawal poochein")
    
    # Bada Mic Button Center mein
    spoken_text = speech_to_text(
        language='ur-PK', 
        start_prompt="🔴 Tap to Speak (Bolein)",
        stop_prompt="⏹️ Sun raha hoon...",
        just_once=True,
        key='STT_VOICE_AGENT'
    )
    
    # Fallback text input (agar mic issue kare)
    written_text = st.chat_input("Ya yahan type karein...")
    
    question = written_text or spoken_text

    if question:
        st.session_state.latest_user_text = question
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.user_queries.append(question)

        # Agent Processing
        with st.spinner("Salman check kar raha hai..."):
            result = agent.invoke({"messages": st.session_state.chat_history})
            final_message = result["messages"][-1].content
            
            st.session_state.latest_response = final_message
            st.session_state.chat_history = list(result["messages"])
            auto_update_summary(st.session_state.session_id, st.session_state.user_queries)
        
        st.rerun()

    # --- UI Display (Subtitles aur Audio) ---
    if st.session_state.latest_user_text:
        st.markdown(f"<p class='subtitle-text'>Aapne pocha: <i>\"{st.session_state.latest_user_text}\"</i></p>", unsafe_allow_html=True)

    st.markdown(f"<p class='salman-reply'>{st.session_state.latest_response}</p>", unsafe_allow_html=True)
    
    # Generate and Auto-play Audio
    audio_data = play_voice(st.session_state.latest_response)
    if audio_data:
        # 1. Background mein auto-play force karne ki trick
        autoplay_audio(audio_data)
        
        # 2. Samne visual audio player (Agar browser phir bhi block kare to user khud Play daba le)
        st.audio(audio_data, format="audio/mp3", autoplay=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2 & 3: Admin Tabs ---
if is_admin:
    with tab_db:
        st.subheader("📊 Showroom Analytics & Database")
        conn = sqlite3.connect(DB_PATH)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Total Tables", value=len(SCHEMA.keys()))
        with m2:
            st.metric(label="System Status", value="Online")
        with m3:
            st.metric(label="Total Chat Sessions", value=st.session_state.session_id)
            
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            selected_table = st.selectbox("📌 Select Table to View:", tables)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
        conn.close()
        
    with tab_logs:
        st.subheader("📝 Live Customer Logs")
        log_conn = sqlite3.connect(LOG_DB_PATH)
        try:
            chat_df = pd.read_sql_query("SELECT session_id, timestamp, summary as user_questions FROM chat_summaries ORDER BY session_id DESC", log_conn)
            st.dataframe(chat_df, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Abhi tak koi chat history nahi hai.")
        finally:
            log_conn.close()
