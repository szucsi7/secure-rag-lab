# 🛡️ Project: Secure-RAG
**A Defensive AI Implementation Lab (OWASP Top 10 for LLM)**

### 🏗️ Architecture
This lab demonstrates a **Defense-in-Depth** architecture for Retrieval-Augmented Generation (RAG). 

### 🔒 Security Features
* **Layer 1: Metadata-Based Whitelisting (ABAC)** - Retrieval is restricted via Vector DB filters. "Employee" roles are blocked from accessing "High" security document chunks.
* **Layer 2: Structural Prompt Tagging** - Uses `<system>`, `<context>`, and `<user>` XML-style delimiters to prevent "Context Smuggling" and maintain instruction hierarchy.
* **Layer 3: Semantic Bouncer (Output Auditor)** - A secondary LLM pass audits the final response for sensitive intent (secrets, project codes) before it reaches the UI.

**Mitigates:** LLM01: Prompt Injection, LLM06: Sensitive Information Disclosure, and Indirect Prompt Injection.

### 🛠️ Tech Stack
* **Orchestration:** LangChain (v0.3+)
* **Database:** ChromaDB (Persistent)
* **Model:** Ollama (Llama 3)
* **Deployment:** Docker Compose

### 🚀 Quick Start
1. **Clone the repo** and navigate to the folder.
2. **Launch the stack:** ```bash docker-compose up -d --build 
                            it will automatically pull the Llama3 model (approx. 5GB)
3. **Access UI:** navigate to url: http://localhost:8501/
4. **Ingestion:** simply drop .txt or .pdf files into the ./knowledge folder
                they will be synchronized automatically on the next app load