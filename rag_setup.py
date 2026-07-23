import pandas as pd
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
import streamlit as st

# Telemetry disable karne ke liye
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools import tool
from typing import Any

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
load_dotenv()

# ==================================================================
# PAGE CONFIGURATION (Streamlit)
# ==================================================================
st.set_page_config(page_title="Shamas Honda - Agent Dashboard", layout="wide", page_icon="🏍️")

DATA_FOLDER = "./data"          # Excel files yahan
DOCS_FOLDER = "./documents"      # PDF/Word files yahan
DB_PATH = "./shamas_honda.db"
PERSIST_FOLDER = "./vectorstore"
LOG_DB_PATH = "./chat_logs.db"   # Logs save karne ke liye database

os.makedirs(DOCS_FOLDER, exist_ok=True)  # agar folder na ho to bana do

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

def save_summary_to_db(session_id, summary_text):
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO chat_summaries (session_id, timestamp, summary) 
            VALUES (?, ?, ?)
        ''', (session_id, timestamp, summary_text))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Log Error] Summary database mein save nahi ho saki: {e}")

# ----------------------------------------------------------------
# CACHE MODELS (Taake Streamlit har click pe models dobara load na kare)
# ----------------------------------------------------------------
@st.cache_resource
def load_models():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return llm, embeddings, reranker

llm, embeddings, reranker_model = load_models()

# ==================================================================
# 3. EXCEL -> SQLite 
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
@tool("search_item_fuzzy")
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

@tool("run_sql_query")
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

# ==================================================================
# 4. PDF / WORD -> Vectorstore
# ==================================================================
def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_docx_text(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(para.text for para in doc.paragraphs)

def load_documents() -> list[Document]:
    docs = []
    for file in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, file)
        if file.endswith(".pdf"):
            text = extract_pdf_text(path)
        elif file.endswith(".docx") and not file.startswith("~$"):
            text = extract_docx_text(path)
        else:
            continue  
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": file}))
    return docs

@st.cache_resource
def get_doc_vectorstore():
    has_files = any(f.endswith((".pdf", ".docx")) for f in os.listdir(DOCS_FOLDER))
    if not has_files:
        return None
    if os.path.exists(PERSIST_FOLDER):
        return Chroma(persist_directory=PERSIST_FOLDER, embedding_function=embeddings)
    
    raw_docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = splitter.split_documents(raw_docs)
    store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=PERSIST_FOLDER)
    return store

doc_vectorstore = get_doc_vectorstore()

@tool("search_documents")
def search_documents(query: str) -> str:
    """PDF ya Word files mein se jawab dhoondne ke liye."""
    if doc_vectorstore is None:
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
# 6. STREAMLIT WEB UI
# ==================================================================
init_logging_db()

# Session State Initialize Karna
if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]
if "display_msgs" not in st.session_state:
    st.session_state.display_msgs = []
if "user_queries" not in st.session_state:
    st.session_state.user_queries = []

# Sidebar (Summary Save karne ke liye)
with st.sidebar:
    st.title("⚙️ Controls")
    st.write(f"**Session ID:** {st.session_state.session_id}")
    if st.button("🔴 End Session & Save Summary"):
        if st.session_state.user_queries:
            with st.spinner("Summary save ho rahi hai..."):
                summary_prompt = f"User ne is session mein yeh sawal pooche hain: {st.session_state.user_queries}. Inki ek choti si summary (1-2 sentences) Roman Urdu mein banao ke user kya dhoond raha tha."
                summary_response = llm.invoke(summary_prompt)
                save_summary_to_db(st.session_state.session_id, summary_response.content)
            st.success("Summary Save Ho Gayi!")
            # Reset Session
            st.session_state.session_id = get_new_session_id()
            st.session_state.chat_history = [SystemMessage(content=system_prompt)]
            st.session_state.display_msgs = []
            st.session_state.user_queries = []
            st.rerun()
        else:
            st.warning("Koi chat nahi hui save karne ke liye.")

st.title("🏍️ Shamas Honda - AI Agent & Dashboard")

# --- PUBLIC CHAT INTERFACE ---
st.subheader("💬 Chat with Salman")

# Pichli chat dikhana
for msg in st.session_state.display_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User ka naya input
if question := st.chat_input("Poochiye Honda CD 70 ki details..."):
    # UI par dikhana
    st.chat_message("user").markdown(question)
    
    # History update karna
    st.session_state.display_msgs.append({"role": "user", "content": question})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.user_queries.append(question)

    # Agent Limits Check
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[-8:]

    # Agent Process Karega
    with st.spinner("Salman check kar raha hai..."):
        result = agent.invoke({"messages": st.session_state.chat_history})
        final_message = result["messages"][-1].content
        
        st.chat_message("assistant").markdown(final_message)
        
        st.session_state.display_msgs.append({"role": "assistant", "content": final_message})
        st.session_state.chat_history = list(result["messages"])

st.divider()

# --- ADMIN PANEL (HIDDEN BEHIND EXPANDER & PASSWORD) ---
with st.expander("🔒 Admin Panel (Sirf Admin Ke Liye)"):
    # Streamlit Cloud par ADMIN_PASSWORD set hona zaroori hai
    # agar abhi set nahi kiya toh .get() app ko crash hone se bachayega
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    
    user_pass = st.text_input("Admin Password enter karein:", type="password")
    
    if user_pass == admin_password and admin_password != "":
        st.success("Welcome! Access Granted.")
        
        # Sahi password par andar wale 2 tabs show honge
        admin_tab1, admin_tab2 = st.tabs(["📊 Database (Excel View)", "📝 Chat Logs & Summaries"])
        
        with admin_tab1:
            st.subheader("📦 Showroom Database")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                selected_table = st.selectbox("Apni Table Select Karein:", tables)
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Abhi tak koi Excel data table maujood nahi hai.")
            conn.close()
            
        with admin_tab2:
            st.subheader("📝 Chat Logs & Summaries")
            log_conn = sqlite3.connect(LOG_DB_PATH)
            try:
                chat_df = pd.read_sql_query("SELECT * FROM chat_summaries", log_conn)
                st.dataframe(chat_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info("Abhi tak koi chat summary save nahi hui.")
            finally:
                log_conn.close()
                
    elif user_pass != "":
        st.error("Ghalat password!")
