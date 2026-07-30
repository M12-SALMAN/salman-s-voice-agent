import pandas as pd
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
import streamlit as st

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
# 1. PAGE CONFIGURATION & ADVANCED THEME
# ==================================================================
st.set_page_config(page_title="Shamas Honda - AI Agent", layout="wide", page_icon="🏍️")

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    /* Main Background aur Font */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Honda Red Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #cc0000 0%, #990000 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(204, 0, 0, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(204, 0, 0, 0.4);
    }
    
    /* Quick Suggestion Outline Buttons */
    div[data-testid="stHorizontalBlock"] button {
        background: transparent !important;
        color: #cc0000 !important;
        border: 2px solid #cc0000 !important;
        box-shadow: none;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #cc0000 !important;
        color: white !important;
    }

    /* Advanced Chat Message Bubbles */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        border: 1px solid #eee;
    }
    
    /* AI Message Marker */
    div[data-testid="chat-message-assistant"] {
        border-left: 5px solid #cc0000;
    }
    /* User Message Marker */
    div[data-testid="chat-message-user"] {
        border-right: 5px solid #1a1a1a;
        background-color: #fcfcfc;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        border: 1px solid #eaeaea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #cc0000 0%, #990000 100%) !important;
        color: white !important;
        border: none;
    }
    
    /* Streamlit ki default branding hide karna */
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
# 2. VOICE GENERATION FUNCTION
# ==================================================================
def play_voice(text):
    try:
        tts = gTTS(text=text, lang='hi')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        print(f"Voice generation error: {e}")
        return None

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
    except Exception as e:
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
    """Kisi bike, part, ya accessory ki detail ke liye (Fuzzy Match)."""
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
# 5. PROMPT + AGENT
# ==================================================================
system_prompt = f"""Tum Shamas Honda, Sialkot ke senior sales dealer ho. Naam: Salman.

Database mein yeh tables maujood hain:
{SCHEMA_TEXT}

Rule 0: HAMESHA pehle tool call karo.
Rule 1: Kisi specific bike, part, ya accessory ki detail/price mangi jaye to HAMESHA 'search_item_fuzzy' tool use karo.
Rule 2: Complex analysis (Counting, Totals) ke liye 'run_sql_query' use karo.
Rule 3: Policy, agreement ya document se related sawal ho to 'search_documents' use karo.
Rule 4: Agar sab tools fail ho jayen, to bolo "Bhai ye info abhi mere paas nahi hai, shop aa kar confirm kar lein".
Rule 5: Jawab Roman Urdu mein do. Friendly aur short rakho. Price ho to Rs likho.
Rule 6: Pichli baatcheet yaad rakho.
Rule 7: Akhir mein poocho "Aur koi help chahiye?"
Rule 8: STRICT RESTRICTION: Apni general knowledge se jawab bilkul nahi dena. Data na mile to mazzrat kar lo.
"""

@st.cache_resource
def get_agent():
    return create_react_agent(model=llm, tools=tools)

agent = get_agent()

# ==================================================================
# 6. STREAMLIT WEB UI & ADMIN LOGIC
# ==================================================================
init_logging_db()

# --- SIDEBAR BRANDING & CONTROLS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda_Logo.svg", width=150)
    st.markdown("## Shamas Honda")
    st.caption("AI Sales & Inventory Agent")
    
    st.divider()
    st.markdown("📍 **Location:** Circular Road, Sialkot")
    st.markdown("📞 **Helpline:** 0300-XXXXXXX")
    st.markdown("⏰ **Timings:** 10:00 AM - 8:00 PM")
    
    st.divider()
    if st.button("🗑️ Nayi Chat Shuru Karein", use_container_width=True):
        for key in ["session_id", "chat_history", "display_msgs", "user_queries"]:
            if key in st.session_state:
                del st.session_state[key]
        st.toast("✅ Nayi Chat Shuru Ho Gayi Hai!", icon="🔄")
        st.rerun()
        
    st.divider()
    
    # --- SECURE ADMIN PANEL ---
    with st.expander("🔒 Admin Access"):
        admin_password = st.secrets.get("ADMIN_PASSWORD", "")
        user_pass = st.text_input("Enter Password", type="password")

is_admin = (user_pass == admin_password and admin_password != "")

# Top Header
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🏍️ Shamas Honda - AI Agent")
with col2:
    if is_admin:
        st.success("Admin Active")

# Session State Initialize
if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()

if "chat_history" not in st.session_state:
    welcome_message = "Assalam o Alaikum! Main Salman baat kar raha hoon Shamas Honda Sialkot se. Main aapki kya madad kar sakta hoon?"
    st.session_state.chat_history = [
        SystemMessage(content=system_prompt),
        AIMessage(content=welcome_message)
    ]
    st.session_state.display_msgs = [
        {"role": "assistant", "content": welcome_message}
    ]
    
if "user_queries" not in st.session_state:
    st.session_state.user_queries = []


if is_admin:
    tab_chat, tab_db, tab_logs = st.tabs(["💬 Chat with Salman", "📊 Admin Dashboard", "📝 Customer Logs"])
else:
    tab_chat = st.container()

# --- TAB 1: Chat Interface ---
with tab_chat:
    
    # Display previous messages
    for msg in st.session_state.display_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === FIXED: Variable pehle define karein ===
    quick_query = None

    # --- Quick Suggestions (Sirf tab nazar ayengi jab user ne koi sawal na pocha ho) ---
    if len(st.session_state.user_queries) == 0:
        st.write("💡 **Quick Suggestions:**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("Honda CD 70 ki price kya hai?", use_container_width=True):
                quick_query = "Honda CD 70 ki price kya hai?"
        with sc2:
            if st.button("CG 125 ke available colors?", use_container_width=True):
                quick_query = "CG 125 ke available colors?"
        with sc3:
            if st.button("Kisto (Installment) ka kya plan hai?", use_container_width=True):
                quick_query = "Kisto (Installment) ka kya plan hai?"

    # --- 🎙️ Mic aur Text Input ---
    col_mic, col_txt = st.columns([0.15, 0.85])
    with col_mic:
        spoken_text = speech_to_text(
            language='ur-PK', 
            start_prompt="🎙️ Bol kar...",
            stop_prompt="🔴 Sun raha hai...",
            just_once=True,
            key='STT'
        )
    with col_txt:
        written_text = st.chat_input("Apna sawal likhein (Maslan: CD 70 ki details...)")
    
    # Input finalization
    question = quick_query or written_text or spoken_text

    if question:
        st.chat_message("user").markdown(question)
        
        st.session_state.display_msgs.append({"role": "user", "content": question})
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.user_queries.append(question)

        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[-8:]

        with st.spinner("Salman system check kar raha hai..."):
            result = agent.invoke({"messages": st.session_state.chat_history})
            final_message = result["messages"][-1].content
            
            st.chat_message("assistant").markdown(final_message)
            
            audio_data = play_voice(final_message)
            if audio_data:
                st.audio(audio_data, format="audio/mp3", autoplay=True)
            
            st.session_state.display_msgs.append({"role": "assistant", "content": final_message})
            st.session_state.chat_history = list(result["messages"])
            
            auto_update_summary(st.session_state.session_id, st.session_state.user_queries)
        
        st.rerun() # Refresh to clear suggestions if used

# --- TAB 2 & 3: Admin Tabs ---
if is_admin:
    with tab_db:
        st.subheader("📊 Showroom Analytics & Database")
        
        # Advanced Stat Cards
        conn = sqlite3.connect(DB_PATH)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Total Tables", value=len(SCHEMA.keys()))
        with m2:
            st.metric(label="System Status", value="Online", delta="Salman is Active")
        with m3:
            st.metric(label="Total Chat Sessions", value=st.session_state.session_id)
            
        st.markdown("---")
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            selected_table = st.selectbox("📌 Select Table to View:", tables)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Abhi tak koi Excel data table maujood nahi hai.")
        conn.close()
        
    with tab_logs:
        st.subheader("📝 Live Customer Logs")
        st.caption("Yeh logs har message ke baad khud update hote hain.")
        log_conn = sqlite3.connect(LOG_DB_PATH)
        try:
            chat_df = pd.read_sql_query("SELECT session_id, timestamp, summary as user_questions FROM chat_summaries ORDER BY session_id DESC", log_conn)
            st.dataframe(chat_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info("Abhi tak koi chat history nahi hai.")
        finally:
            log_conn.close()
