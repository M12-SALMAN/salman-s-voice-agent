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

DATA_FOLDER = "./data"          
DOCS_FOLDER = "./documents"      
DB_PATH = "./shamas_honda.db"
PERSIST_FOLDER = "./vectorstore"
LOG_DB_PATH = "./chat_logs.db"   

os.makedirs(DOCS_FOLDER, exist_ok=True)  

# ==================================================================
# LOGGING DATABASE SETUP (Auto-Save Logic)
# ==================================================================
def init_logging_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    # 'session_id' PRIMARY KEY hai, is liye hum ek hi session ko bar bar update kar sakte hain
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
    """Yeh function har message ke baad khud ba khud database update karega"""
    if not user_queries_list:
        return
        
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # User ne jo bhi sawal pooche hain unko ek text mein mila kar save kar dega
        summary_text = " | ".join(user_queries_list)
        
        # INSERT OR REPLACE ka faida ye hai ke purani entry update ho jati hai
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
# 6. STREAMLIT WEB UI & ADMIN LOGIC
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

# --- SECURE ADMIN SIDEBAR ---
with st.sidebar:
    st.title("🔒 System Access")
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    user_pass = st.text_input("Enter Password", type="password")

# Check if admin is logged in
is_admin = (user_pass == admin_password and admin_password != "")

st.title("🏍️ Shamas Honda - AI Agent")

# Agar Admin logged in hai, toh Tabs dikhayein. Warna sirf Chat container banayein.
if is_admin:
    st.success("Admin Logged In! Dashboard and Logs Unlocked.")
    tab_chat, tab_db, tab_logs = st.tabs(["💬 Chat", "📊 Database", "📝 Customer Logs"])
else:
    # Aam user ke liye koi tabs nahi banenge, sirf ek sada container hoga
    tab_chat = st.container()

# --- TAB 1: Chat Interface (Sab ke liye visible) ---
with tab_chat:
    st.subheader("💬 Chat with Salman")

    for msg in st.session_state.display_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Poochiye Honda CD 70 ki details..."):
        st.chat_message("user").markdown(question)
        
        st.session_state.display_msgs.append({"role": "user", "content": question})
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.user_queries.append(question)

        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[-8:]

        with st.spinner("Salman check kar raha hai..."):
            result = agent.invoke({"messages": st.session_state.chat_history})
            final_message = result["messages"][-1].content
            
            st.chat_message("assistant").markdown(final_message)
            st.session_state.display_msgs.append({"role": "assistant", "content": final_message})
            st.session_state.chat_history = list(result["messages"])
            
            # --- AUTO SAVE MAGIC (Bina kisi button ke) ---
            auto_update_summary(st.session_state.session_id, st.session_state.user_queries)

# --- TAB 2 & 3: Admin Tabs (Sirf admin ke liye visible) ---
if is_admin:
    with tab_db:
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
