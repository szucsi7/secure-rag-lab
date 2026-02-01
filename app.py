import streamlit as st
import ollama
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETTINGS & TITLE ---
st.set_page_config(page_title="LLM Security Lab", layout="wide")
st.title("🛡️ Secure RAG Lab")

# --- CONFIG ---
KNOWLEDGE_DIR = "./knowledge"
PERSIST_DIR = "/app/chroma_data"
DB_COLLECTION = "architect_lab"

# --- 2. CACHED RESOURCES ---
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="llama3", base_url="http://ollama-architect:11434")

# --- 3. THE CORE ENGINE ---
def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=DB_COLLECTION
    )

def sync_knowledge_base(vector_db):
    # Ingestion: Only processes files that aren't already in the DB.
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return []

    # Get Disk state vs DB state
    current_files = [os.path.join(KNOWLEDGE_DIR, f) for f in os.listdir(KNOWLEDGE_DIR) 
                     if f.endswith(('.txt', '.pdf'))]
    
    existing_data = vector_db.get()
    indexed_sources = set([m.get('source') for m in existing_data['metadatas']]) if existing_data['metadatas'] else set()

    files_to_process = [f for f in current_files if f not in indexed_sources]

    if not files_to_process:
        return []

    new_docs = []
    for file_path in files_to_process:
        st.sidebar.info(f"📥 Ingesting: {os.path.basename(file_path)}")
        
        try:
            loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path, autodetect_encoding=True)
            loaded_docs = loader.load()
            
            # Metadata Tagging
            for d in loaded_docs:
                d.metadata['security_level'] = 'high' if any(x in file_path.lower() for x in ["financial", "secret"]) else 'low'
            new_docs.extend(loaded_docs)
        except Exception as e:
            st.sidebar.error(f"❌ Error: {file_path}")

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(new_docs)
        vector_db.add_documents(chunks)
        return files_to_process
    
    return []

# --- 4. INITIALIZE APP ---
vectorstore = get_vectorstore()
newly_added = sync_knowledge_base(vectorstore)

# Sidebar UI
with st.sidebar:
    # Usr toggle
    st.header("Admin Settings")
    user_role = st.selectbox("Identity Profile:", ["Employee", "Admin"])

    # DB current status
    st.divider()
    st.subheader("📊 Database Stats")
    # New ingestion stats 
    if newly_added:
        st.success(f"Success! {len(newly_added)} files added.")
    else:
        st.info("Database synchronized.")

    if st.button("List Loaded Files"):
        data = vectorstore.get()
        st.write(f"Total Chunks in DB: {len(data['ids'])}")
        
        # List all unique source files found in the DB
        sources = set([m.get('source') for m in data['metadatas']])
        st.write("Indexed Files:")
        for s in sources:
            st.write(s)
        
    if st.button("🔥 Purge & Re-index"):
        os.system(f"rm -rf {PERSIST_DIR}/*")
        st.rerun()


# --- 5. SECURITY LAYERS (The Bouncer) ---
def is_response_safe(user_query, ai_response):
    bouncer_prompt = f"""
    [SYSTEM] You are a Security Auditor. Respond ONLY with 'SAFE' or 'UNSAFE'.
    [CONTEXT] User asked: {user_query} | AI responded: {ai_response}
    [RULES] Mark UNSAFE if:
    - Mentions internal passwords or project codes.
    - Mentions bankruptcy, frozen budgets, or acquisition secrets.
    """
    check = ollama.generate(model="llama3", prompt=bouncer_prompt)
    return "SAFE" in check['response'].upper()

# --- 6. USER QUERY LOGIC ---
query = st.text_input("Ask the Corporate Assistant:")

if query:
    # 1. RETRIEVAL (Role-Based Filter)
    search_kwargs = {"k": 3}
    if user_role != "Admin":
        search_kwargs["filter"] = {"security_level": "low"}
    
    results = vectorstore.similarity_search(query, **search_kwargs)
    
    if not results:
        st.error("Access Denied or No Information Found.")
    else:
        context = results[0].page_content
        
        # 2. GENERATION (The Model Inference)
        prompt = f"<system>You are a corporate assistant.</system><context>{context}</context><user>{query}</user>"
        response = ollama.generate(model="llama3", prompt=prompt)
        raw_response = response['response']

        # 3. THE BOUNCER (The Final Semantic Check)
        if is_response_safe(query, raw_response):
            st.subheader("Assistant Response:")
            st.write(raw_response)
        else:
            st.error("🚨 SECURITY ALERT: The Bouncer blocked a suspicious response.")
            st.info("The model attempted to discuss restricted financial status or secrets.")

        # 4 AUDIT LOG
        with st.expander("🔍 Architect Audit Log"):
            for idx, doc in enumerate(results):
                st.markdown(f"**Result {idx+1} Source:** {doc.metadata.get('source')}")
                st.write(f"**Security Level:** {doc.metadata.get('security_level')}")
                st.info(doc.page_content)
                st.divider()