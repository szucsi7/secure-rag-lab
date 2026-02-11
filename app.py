import streamlit as st
import ollama
import os
import shutil
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 1. CONFIGURATION & DOMAIN OBJECTS ---

@dataclass
class AppConfig:
    PAGE_TITLE: str = "🛡️ Secure RAG Lab"
    KNOWLEDGE_DIR: str = "./knowledge"
    PERSIST_DIR: str = "/app/chroma_data"
    DB_COLLECTION: str = "architect_lab"
    OLLAMA_BASE_URL: str = "http://ollama-architect:11434"
    EMBEDDING_MODEL: str = "llama3"
    GENERATION_MODEL: str = "llama3"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

class SecurityLevel:
    HIGH = "high"
    LOW = "low"

# --- 2. SERVICE LAYER ---

class KnowledgeBase:
    """Handles all Vector Database interactions and Document Ingestion."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization of the vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.config.PERSIST_DIR,
                embedding_function=OllamaEmbeddings(
                    model=self.config.EMBEDDING_MODEL,
                    base_url=self.config.OLLAMA_BASE_URL
                ),
                collection_name=self.config.DB_COLLECTION
            )
        return self._vectorstore

    def reset_database(self) -> None:
        """Purges the database."""
        try:
            # Chroma client reset
            # Accessing private client as per original implementation pattern
            # In production, might prefer deleting the persist dir, but respecting original logic:
            self.vectorstore._client.reset()
            self._vectorstore = None
        except Exception as e:
            raise RuntimeError(f"Database reset failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns database statistics."""
        data = self.vectorstore.get()
        return {
            "total_chunks": len(data['ids']) if data else 0,
            "sources": set(m.get('source') for m in data['metadatas']) if data and data['metadatas'] else set()
        }

    def sync_files(self) -> List[str]:
        """Ingests new files from the knowledge directory."""
        if not os.path.exists(self.config.KNOWLEDGE_DIR):
            os.makedirs(self.config.KNOWLEDGE_DIR)
            return []

        # Identify files
        disk_files = [
            os.path.join(self.config.KNOWLEDGE_DIR, f) 
            for f in os.listdir(self.config.KNOWLEDGE_DIR) 
            if f.endswith(('.txt', '.pdf'))
        ]
        
        # Identify what's already indexed
        db_stats = self.get_stats()
        indexed_sources = db_stats["sources"]

        files_to_process = [f for f in disk_files if f not in indexed_sources]
        
        if not files_to_process:
            return []

        new_docs = self._load_documents(files_to_process)
        if new_docs:
            self._index_documents(new_docs)
            
        return files_to_process

    def _load_documents(self, file_paths: List[str]) -> List[Document]:
        docs = []
        for file_path in file_paths:
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, autodetect_encoding=True)
                
                loaded = loader.load()
                # Metadata Tagging
                is_sensitive = any(keyword in file_path.lower() for keyword in ["financial", "secret"])
                for d in loaded:
                    d.metadata['security_level'] = SecurityLevel.HIGH if is_sensitive else SecurityLevel.LOW
                docs.extend(loaded)
            except Exception as e:
                # In a real app, use proper logging
                st.sidebar.error(f"❌ Error loading {os.path.basename(file_path)}: {e}")
        return docs

    def _index_documents(self, docs: List[Document]) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE, 
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

class SecurityAgent:
    """The 'Bouncer': Handles output validation and safety checks."""
    
    def __init__(self, config: AppConfig):
        self.model = config.GENERATION_MODEL

    def is_response_safe(self, user_query: str, ai_response: str) -> bool:
        """Uses an LLM to audit the response."""
        bouncer_prompt = f"""
        [SYSTEM] You are a Security Auditor. Respond ONLY with 'SAFE' or 'UNSAFE'.
        [CONTEXT] User asked: {user_query} | AI responded: {ai_response}
        [RULES] Mark UNSAFE if:
        - Mentions internal passwords or project codes.
        - Mentions bankruptcy, frozen budgets, or acquisition secrets.
        """
        try:
            check = ollama.generate(model=self.model, prompt=bouncer_prompt)
            return "SAFE" in check['response'].upper()
        except Exception:
            # Fail closed (secure) if the check fails
            return False

class RAGEngine:
    """Orchestrates Retrieval and Generation."""
    
    def __init__(self, knowledge_base: KnowledgeBase, config: AppConfig):
        self.kb = knowledge_base
        self.config = config

    def query(self, user_query: str, user_role: str) -> Dict[str, Any]:
        """
        Executes the RAG pipeline.
        Returns a dict containing the raw response, context, and safety status.
        """
        # 1. Retrieval
        search_kwargs = {"k": 3}
        if user_role != "Admin":
            search_kwargs["filter"] = {"security_level": SecurityLevel.LOW}
        
        results = self.kb.vectorstore.similarity_search(user_query, **search_kwargs)
        
        if not results:
            return {"found": False, "message": "Access Denied or No Information Found."}

        # 2. Context Construction
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # 3. Generation
        prompt = (
            f"<system>You are a corporate assistant. Use the provided context to answer.</system>"
            f"<context>{context_text}</context>"
            f"<user>{user_query}</user>"
        )
        
        response_obj = ollama.generate(model=self.config.GENERATION_MODEL, prompt=prompt)
        raw_response = response_obj['response']

        return {
            "found": True,
            "raw_response": raw_response,
            "context_docs": results
        }

# --- 3. UI LAYER (Streamlit) ---

def initialize_session(kb: KnowledgeBase):
    """Handles session state initialization and one-time sync."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.newly_added = kb.sync_files()

def render_sidebar(kb: KnowledgeBase):
    with st.sidebar:
        st.header("Admin Settings")
        role = st.selectbox("Identity Profile:", ["Employee", "Admin"])
        
        st.divider()
        st.subheader("📊 Database Stats")
        
        # Show sync result from session
        if st.session_state.get("newly_added"):
             st.success(f"Synced {len(st.session_state.newly_added)} new files.")
        else:
            st.info("Database synchronized.")

        if st.button("List Loaded Files"):
            stats = kb.get_stats()
            st.write(f"Total Chunks: {stats['total_chunks']}")
            st.write("Indexed Files:")
            for s in stats['sources']:
                st.write(f"- {os.path.basename(s)}")

        if st.button("🔥 Purge & Re-index"):
            try:
                kb.reset_database()
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Reset Complete. Reloading...")
                st.rerun()
            except Exception as e:
                st.error(f"Purge failed: {e}")
                
        return role

def render_main(rag: RAGEngine, bouncer: SecurityAgent, role: str):
    query = st.text_input("Ask the Corporate Assistant:")
    
    if query:
        with st.spinner("Analyzing knowledge base..."):
            result = rag.query(query, role)

        if not result["found"]:
            st.error(result["message"])
            return

        raw_response = result["raw_response"]
        
        # The Bouncer Check
        if bouncer.is_response_safe(query, raw_response):
            st.subheader("Assistant Response:")
            st.write(raw_response)
        else:
            st.error("🚨 SECURITY ALERT: The Bouncer blocked a suspicious response.")
            st.info("The model attempted to discuss restricted financial status or secrets.")

        # Audit Log
        with st.expander("🔍 Audit Log"):
            st.markdown(f"**User Role:** {role}")
            st.markdown("**Retrieved Context:**")
            for idx, doc in enumerate(result["context_docs"]):
                st.markdown(f"**Source {idx+1}:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                st.markdown(f"**Security Level:** `{doc.metadata.get('security_level')}`")
                st.caption(doc.page_content[:200] + "...")
                st.divider()

# --- 4. APPLICATION ENTRY POINT ---

def main():
    config = AppConfig()
    st.set_page_config(page_title=config.PAGE_TITLE, layout="wide")
    st.title(config.PAGE_TITLE)

    # Dependency Injection
    # We use st.cache_resource for the KnowledgeBase to persist the connection across re-runs
    # but the class itself is lightweight wrapper around the cached vectorstore
    
    kb = KnowledgeBase(config)
    bouncer = SecurityAgent(config)
    rag = RAGEngine(kb, config)

    initialize_session(kb)
    
    user_role = render_sidebar(kb)
    render_main(rag, bouncer, user_role)

if __name__ == "__main__":
    main()