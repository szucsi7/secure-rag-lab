# 🛡️ Project: Secure-RAG
**A Defensive AI Implementation Lab (OWASP Top 10 for LLM)**

This lab demonstrates a **Defense-in-Depth** architecture for Retrieval-Augmented Generation (RAG).

### 🏗️ Architecture
* **Hybrid Retrieval (Ensemble):** Combines **Vector Embeddings (Chroma)** with **Keyword Ranking (BM25)** using a 0.4/0.6 weighted split. This ensures that specific texts are caught even if semantic meaning is ambiguous.
* **Hardened Containerization:**
    * **Least Privilege:** The application runs as a non-root `appuser` to mitigate container breakout risks.

### 🔒 Security Features
* **Layer 1: Metadata-Based Whitelisting (ABAC)** - Strict whitelisting via Vector DB filters. "Employee" roles are physically blocked from the search index for "High" security chunks.
* **Layer 2: Structural Prompt Tagging** - Uses `<system>`, `<context>`, and `<user>` XML-style delimiters to prevent "Context Smuggling" and maintain instruction hierarchy.
* **Layer 3: Semantic Bouncer (Output Auditor)** - A secondary LLM pass (SecurityAgent) audits the final response for sensitive intent (secrets, project codes) before it reaches the UI.

**Mitigates:** LLM01: Prompt Injection, LLM06: Sensitive Information Disclosure, and Indirect Prompt Injection.

### 🛠️ Tech Stack
* **Orchestration:** LangChain & LangChain-Community 
* **Database:** ChromaDB 
* **Model:** Ollama (Llama 3) with NVIDIA GPU Acceleration
* **Search Logic:** `rank_bm25` (Reciprocal Rank Fusion)
* **Deployment:** Docker Compose (Service-Healthy Dependencies)

### 🚀 Quick Start
1. **Clone the repo** and navigate to the folder.
2. **Launch the stack:** ```bash docker-compose up -d --build 
                            it will automatically pull the Llama3 model (approx. 5GB)
3. **Access UI:** navigate to url: http://localhost:8501/
4. **Ingestion:** simply drop .txt or .pdf files into the ./knowledge folder
                they will be synchronized automatically on the next app load              