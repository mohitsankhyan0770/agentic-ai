# agentic-ai
# Agentic AI System (Multi-Agent LLM)

An autonomous AI system built using local Large Language Models (LLMs) that performs intelligent task execution through planning, memory, tool usage, and self-reflection.

This project demonstrates a modular agent architecture capable of breaking down tasks, executing them using tools, improving responses, and learning from past interactions.


# Features

- Planning module to break complex user queries into structured steps  
- Tool-using agent capable of executing calculator, file operations, and search  
- Memory system using vector embeddings for contextual recall (ChromaDB)  
- Self-critic module to evaluate and improve response quality  
- Local LLM execution using Ollama (no external API dependency)  


# Tech Stack

- Python  
- Ollama (Local LLM runtime)  
- ChromaDB (Vector database)  
- Sentence Transformers (Embeddings)  
- JSON-based tool orchestration  

# System Architecture

User Input  
→ Planner (creates steps)  
→ Agent (decides action/tool)  
→ Tool Execution or Direct Response  
→ Critic (evaluates output)  
→ Memory Storage  
→ Final Output  


## Project Structure

agentic-ai-system/  
│── agent.py        # Core agent logic  
│── planner.py      # Task planning  
│── critic.py       # Response evaluation  
│── memory.py       # Vector memory system  
│── test_llm.py     # LLM test script  
│── README.md  


# Setup and Run

1. Install dependencies:

pip install ollama chromadb sentence-transformers

2. Install Ollama:

https://ollama.com

3. Pull model:

ollama pull phi3

4. Run the agent:

python agent.py


# Capabilities

- Solves mathematical expressions using tools  
- Stores and recalls previous interactions  
- Breaks complex problems into steps  
- Improves responses using self-critique  
- Executes multi-step reasoning workflows  


# Future Improvements

- Multi-agent collaboration  
- Web interface (Flask or React)  
- External API integrations  
- Task automation pipelines  
- Cloud deployment  

