# Agent Development Kit (ADK) Crash Course - Project Documentation

**Last Updated:** November 16, 2025
**Repository:** `sphilius/agent-development-kit-crash-course`
**Current Branch:** `claude/add-project-documentation-01LehhSHFNJKS3tZmZUBMz72`

---

## Project Overview

This is an **educational repository** designed as a comprehensive crash course for learning **Google's Agent Development Kit (ADK)**, a framework for building LLM-powered agents using Gemini models. The repository contains 12 progressive tutorial examples plus a custom **AuDHD RAG Agent** as a capstone project.

### Key Metrics
- **Total Examples:** 12 progressive learning examples
- **Custom Project:** 1 RAG-based agent (auhdhd_rag_agent)
- **Total Python Files:** 77
- **Repository Size:** 720KB
- **Status:** Clean (no uncommitted changes)

### Primary Purpose
Provide a structured learning path from basic to advanced ADK concepts, enabling developers to build sophisticated LLM-powered agents with tools, state management, multi-agent orchestration, and iterative refinement loops.

---

## Architecture & Project Structure

### Tutorial Examples (1-12): Progressive Learning Path

#### **Example 1: Basic Agent**
- **Location:** `1-basic-agent/`
- **Purpose:** Foundation - demonstrates simplest ADK agent implementation
- **Key Concepts:** Agent initialization, naming, descriptions, instructions
- **Example:** Greeting agent
- **Status:** ✅ Complete

#### **Example 2: Tool Agent**
- **Location:** `2-tool-agent/`
- **Purpose:** Teach how agents use tools for extended capabilities
- **Key Concepts:**
  - Built-in tools: Google Search, Code Execution, Vertex AI Search
  - Custom function tools
- **Important Limitation:** Only one built-in tool per agent; cannot mix built-in tools with custom functions
- **Example:** Multi-tool agent
- **Status:** ✅ Complete

#### **Example 3: LiteLLM Agent**
- **Location:** `3-litellm-agent/`
- **Purpose:** Show model abstraction and provider flexibility
- **Key Concepts:** LiteLLM integration for switching between LLM providers (OpenAI, Anthropic, etc.)
- **Example:** Dad joke generator using alternative LLM provider
- **Status:** ✅ Complete

#### **Example 4: Structured Outputs**
- **Location:** `4-structured-outputs/`
- **Purpose:** Ensure consistent, predictable JSON responses
- **Key Concepts:** Pydantic models with `output_schema` parameter
- **Important Limitation:** Cannot use tools when output_schema is defined
- **Example:** Email generator with structured subject/body fields
- **Status:** ✅ Complete

#### **Example 5: Sessions and State**
- **Location:** `5-sessions-and-state/`
- **Purpose:** Maintain conversation history and context across interactions
- **Key Concepts:** Session management, state persistence across turns
- **Example:** Question answering agent with memory
- **Status:** ✅ Complete

#### **Example 6: Persistent Storage**
- **Location:** `6-persistent-storage/`
- **Purpose:** Demonstrate data persistence across application restarts
- **Key Concepts:** SQLite database integration, permanent data storage
- **Example:** Reminder/todo agent with SQLite backend
- **Status:** ✅ Complete

#### **Example 7: Multi-Agent System**
- **Location:** `7-multi-agent/`
- **Purpose:** Orchestrate multiple specialized agents working together
- **Key Concepts:**
  - Manager agent pattern
  - AgentTool for sub-agent delegation
  - Specialized sub-agents
- **Example:** Manager agent with 3 sub-agents (stock_analyst, funny_nerd, news_analyst)
- **Status:** ✅ Complete

#### **Example 8: Stateful Multi-Agent System**
- **Location:** `8-stateful-multi-agent/`
- **Purpose:** Combine multi-agent orchestration with shared state management
- **Key Concepts:**
  - Multi-agent with persistent sessions
  - Shared state across sub-agents
  - Specialized agent roles
- **Example:** Customer service system with agents for course support, orders, policy, and sales
- **Architecture:**
  - Manager agent coordinates sub-agents
  - Session state: user_name, purchased_courses, interaction_history
  - Sub-agents access and modify shared state
- **Status:** ✅ Complete

#### **Example 9: Callbacks & Monitoring**
- **Location:** `9-callbacks/`
- **Purpose:** Monitor and hook into agent behavior in real-time
- **Key Concepts:**
  - Before/after callbacks
  - Event monitoring for agents, models, tools
  - Real-time behavior tracking
- **Status:** ✅ Complete (recently cleaned up)

#### **Example 10: Sequential Agent Pipeline**
- **Location:** `10-sequential-agent/`
- **Purpose:** Define workflows with strict sequential execution
- **Key Concepts:**
  - Pipeline pattern
  - Ordered agent execution
  - Data flow through pipeline stages
- **Example:** Lead qualification (validator → scorer → recommender)
- **Status:** ✅ Complete

#### **Example 11: Parallel Agent Execution**
- **Location:** `11-parallel-agent/`
- **Purpose:** Execute multiple agents concurrently for performance
- **Key Concepts:**
  - Concurrent agent execution
  - Parallel data collection
  - Result synthesis
- **Example:** System monitor with parallel sub-agents (CPU, disk, memory info)
- **Status:** ✅ Complete

#### **Example 12: Loop Agent with Refinement**
- **Location:** `12-loop-agent/`
- **Purpose:** Iterative refinement through feedback loops
- **Key Concepts:**
  - Loop control with `exit_loop` tool
  - Iterative improvement
  - Quality validation and feedback
- **Example:** LinkedIn post generator with iterative quality checking
- **Configuration:** Max iterations: 10
- **Status:** ✅ Complete

---

### Custom Project: AuDHD RAG Agent

**Location:** `auhdhd_rag_agent/`
**Status:** ✅ Functional (Docker-ready, Pinecone-integrated)
**Last Commit:** c6f555c (Merge PR #1)

#### Purpose
An AuDHD-friendly (Autism & ADHD) Retrieval-Augmented Generation (RAG) agent that answers questions based on a knowledge base using Claude models via OpenRouter.

#### Architecture

**1. Agent Layer** (`agent.py`)
- **Model:** Claude 3 Haiku via OpenRouter (`openrouter/anthropic/claude-3-haiku-20240307`)
- **Framework:** Google ADK with LiteLLM
- **Tool:** `query_knowledge_base` for retrieval
- **Design:** Clear, explicit responses tailored for neurodivergent users

**2. Retrieval System** (`retrieve_knowledge_tool.py`)
- **Backend Options:**
  - **Cloud (Pinecone):** Scalable vector database with OpenAI embeddings
  - **Local (FAISS):** In-process vector store with HuggingFace embeddings
- **Embedding Model:** text-embedding-ada-002 (OpenAI)
- **Configuration:**
  - Top-K results: 3
  - Chunk size: 500-1000 characters with overlap
- **Environment Variables:**
  - `OPENAI_API_KEY` - for embeddings
  - `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME` - for cloud backend
  - `OPENROUTER_API_KEY` - for Claude model access

**3. Data Ingestion**
- **Local Ingestion** (`ingest.py`):
  - Uses FAISS for vector storage
  - HuggingFace embeddings
  - File-based knowledge base
- **Cloud Ingestion** (`ingest_cloud.py`):
  - Syncs with Pinecone
  - OpenAI embeddings
  - Persistent cloud storage

**4. CLI Interface** (`main.py`)
- Interactive command-line agent
- ADK Runner integration
- Session-aware conversation

**5. Docker Support**
- `Dockerfile`: Python 3.10-slim base
- `docker-compose.yml`: Service orchestration
- Port: 8080
- Environment file integration (`.env`)

#### Current Status
- ✅ Basic agent structure implemented
- ✅ Docker containerization complete
- ✅ Pinecone integration working
- ✅ Multiple ingestion options available
- ❌ Missing `__init__.py` for ADK discovery via `adk web`
- ❌ Knowledge base minimal (5 sample sentences in `knowledge.txt`)
- ❌ No comprehensive README for this sub-project

#### Dependencies
```
google-adk[database]==0.3.0
litellm==1.66.3
python-dotenv==1.1.0
openai==1.37.0
pinecone-client==4.2.0
langchain==0.2.11
langchain-openai==0.1.17
langchain-pinecone==0.1.2
ipykernel
```

---

## What's Been Accomplished

### ✅ Core Teaching Framework
- **12 complete, well-documented examples** covering the full spectrum of ADK capabilities
- Progressive complexity from basic to advanced patterns
- Each example includes:
  - Focused implementation demonstrating one or more concepts
  - README.md with explanations
  - Runnable example code
  - Clear comments and structure

### ✅ Technology Integration
- **Google ADK Integration:** Complete ADK usage across all examples
- **Gemini Models:** Support for multiple Gemini model families
- **LiteLLM Support:** Added in Example 3, used in RAG agent
- **Multiple LLM Providers:** OpenRouter, OpenAI, Google Gemini

### ✅ Advanced Patterns
- Multi-agent orchestration (Examples 7, 8)
- Stateful agent systems (Examples 5, 6, 8)
- Sequential and parallel workflows (Examples 10, 11)
- Iterative refinement loops (Example 12)
- Callback/monitoring infrastructure (Example 9)

### ✅ Practical Implementation (RAG Agent)
- Docker containerization for easy deployment
- Cloud vector database integration (Pinecone)
- Local alternative implementation (FAISS)
- Multiple LLM provider support
- Environment-based configuration

### ✅ Documentation
- Individual README files for complex examples
- `GEMINI_MODELS.md`: Comprehensive model selection guide
- Inline code comments and explanations
- Git history with descriptive commits

### ✅ DevOps & Deployment
- Docker support for containerization
- docker-compose for service orchestration
- Environment configuration via `.env`
- Clean `.gitignore` setup

---

## What's Worked Well

### 1. **Educational Structure**
- The 12-example progression is effective for learning
- Each example isolates a specific concept or pattern
- Users can understand complexity incrementally

### 2. **Real-World Applicability**
- Examples aren't just tutorials—they demonstrate production-ready patterns
- Multi-agent and stateful systems are genuinely useful
- Callback system teaches monitoring best practices

### 3. **Tool Capabilities**
- Comprehensive coverage of ADK tools (Google Search, Code Execution, Vertex AI)
- Custom function tools clearly demonstrated
- Tool integration patterns well-established

### 4. **Integration Points**
- LiteLLM integration (Example 3) enables multi-provider support
- OpenRouter integration in RAG agent allows access to non-Google models
- Flexible architecture supports both Google and third-party providers

### 5. **Code Quality**
- Clean, readable implementations
- Consistent naming conventions
- Minimal technical debt (recent cleanup commits visible)

---

## What Hasn't Worked / Known Issues

### 1. **RAG Agent Incomplete Integration**
- **Issue:** Missing `__init__.py` file prevents ADK discovery via `adk web`
- **Impact:** Agent cannot be served as a web service through ADK's standard discovery mechanism
- **Fix Required:** Add `__init__.py` with proper ADK structure

### 2. **Minimal Knowledge Base**
- **Issue:** `knowledge.txt` contains only 5 sample sentences
- **Impact:** RAG agent cannot demonstrate real-world RAG effectiveness
- **Fix Required:** Populate with substantive content (50+ relevant sentences/documents)

### 3. **Incomplete Documentation for RAG Project**
- **Issue:** No comprehensive README explaining setup, configuration, and usage
- **Impact:** Users cannot easily understand how to use or extend the RAG agent
- **Fix Required:** Create detailed README with setup steps, environment variables, and examples

### 4. **Tool Limitation Unclear**
- **Issue:** Example 2 notes that only one built-in tool can be used per agent, but this limitation isn't well-documented across other examples
- **Impact:** Users might attempt impossible configurations
- **Fix Required:** Clearer documentation of ADK constraints in central README

### 5. **No Integration Tests**
- **Issue:** No automated tests for agent functionality
- **Impact:** Changes might break example functionality silently
- **Fix Required:** Add basic test suite covering example execution

### 6. **Missing Setup Scripts**
- **Issue:** Commit message mentions setup.sh and setup.ps1, but they're not in current directory
- **Impact:** Setup is more manual than intended
- **Fix Required:** Create or restore setup automation scripts

---

## Development History

### Recent Major Changes

| Commit | Message | Changes |
|--------|---------|---------|
| c6f555c | Merge PR #1: Dockerize application & add setup scripts | +575 lines: Docker files, agent, ingestion, retrieval tool |
| ab64b9b | example 9 cleanup | Cleaned up callbacks example |
| 67d384d | drop unnecessary code | Simplified utils.py in stateful multi-agent |
| e36cbc6 | clean up and improve stateful multi agent | Enhanced customer service agent implementation |

### Key Insights from Git History
- Recent focus on **Docker containerization** for deployment
- Active **refactoring and cleanup** showing code quality focus
- **Feature-branch workflow** with PRs for major changes
- Commit messages follow **conventional commits** style

---

## What's Next / Recommended Actions

### **High Priority**

1. **Fix RAG Agent ADK Integration**
   - Add `__init__.py` to auhdhd_rag_agent/
   - Update agent structure for ADK discovery
   - **Effort:** 30 minutes
   - **Impact:** Enable `adk web` deployment

2. **Expand Knowledge Base**
   - Populate knowledge.txt with 50+ substantial entries
   - Consider organizing by topics/categories
   - **Effort:** 1-2 hours
   - **Impact:** Demonstrate real RAG effectiveness

3. **Create RAG Agent README**
   - Setup instructions
   - Environment variable configuration guide
   - Usage examples
   - Troubleshooting section
   - **Effort:** 1-2 hours
   - **Impact:** Enable self-service user adoption

### **Medium Priority**

4. **Add Project-Level Documentation**
   - Central CLAUDE.md (this document)
   - Architecture overview diagram (ASCII or visual)
   - Quick start guide
   - **Effort:** 2-3 hours
   - **Impact:** Improve first-time user experience

5. **Create Setup Automation**
   - setup.sh for Linux/Mac
   - setup.ps1 for Windows
   - Virtual environment creation
   - Dependency installation
   - **Effort:** 1 hour
   - **Impact:** Reduce setup friction

6. **Add Test Coverage**
   - Example runner tests (verify each example works)
   - RAG agent unit tests
   - CI/CD integration
   - **Effort:** 3-4 hours
   - **Impact:** Catch regressions early

### **Lower Priority**

7. **Documentation Enhancements**
   - Video tutorials linking to examples
   - API reference for ADK patterns
   - Troubleshooting guide
   - Performance optimization tips

8. **Code Quality**
   - Add type hints (currently minimal)
   - Consistent logging across examples
   - Error handling improvements
   - **Effort:** 2-3 hours

9. **Example Expansion**
   - Vision/multimodal agents
   - Agents with memory systems
   - Advanced prompt engineering patterns
   - **Effort:** 2-4 hours per example

---

## Technology Stack

### Core Dependencies
```
google-adk[database]==0.3.0          # Agent framework
google-generativeai==0.8.5           # Gemini API access
litellm==1.66.3                      # Multi-provider LLM abstraction
```

### LLM Integration
- **Google Gemini:** Examples 1-2, 5-12 (native ADK)
- **OpenRouter + Claude:** RAG agent (via LiteLLM)
- **OpenAI:** Available via LiteLLM

### Data & Storage
```
python-dotenv==1.1.0                 # Environment configuration
pinecone-client==4.2.0               # Cloud vector database
openai==1.37.0                       # Embeddings API
langchain==0.2.11                    # RAG framework
langchain-openai==0.1.17
langchain-pinecone==0.1.2
```

### Utilities & Monitoring
```
yfinance==0.2.56                     # Financial data (Example 7)
psutil==5.9.5                        # System info (Example 11)
```

### Deployment
- **Docker:** Python 3.10-slim base image
- **Orchestration:** docker-compose.yml for service management

---

## Setup & Configuration

### Initial Setup
1. Clone repository
2. Create Python 3.9+ virtual environment
3. Install root dependencies: `pip install -r requirements.txt`
4. Create `.env` files in example directories with API keys

### Environment Variables Required
**For Google Examples:**
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)

**For RAG Agent:**
- `OPENROUTER_API_KEY` (Claude access)
- `OPENAI_API_KEY` (embeddings)
- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME` (cloud vector DB)

### Running Examples
```bash
# Navigate to example directory
cd 1-basic-agent

# Run the agent
python -m greeting_agent.agent
```

### Running RAG Agent
```bash
# Docker approach (recommended)
cd auhdhd_rag_agent
docker-compose up

# Or local with Pinecone
python main.py
```

---

## Key Learnings & Constraints

### ADK-Specific Constraints
1. **One Built-in Tool Per Agent:** Cannot use multiple Google built-in tools in single agent
2. **Output Schema Incompatibility:** When using `output_schema` for structured output, tools cannot be used
3. **Tool Execution Model:** Tools run with specific permissions and contexts

### LLM Provider Notes
- **Gemini Models:** Best for ADK integration, full capability support
- **OpenAI/Other Models:** Require LiteLLM wrapper, some feature limitations
- **Model Selection:** Use `gemini-2.0-flash` for balanced performance/cost

### Database & Storage
- **SQLite:** Good for single-instance persistence (Example 6)
- **Pinecone:** Preferred for cloud-based RAG applications
- **FAISS:** Good for local, development RAG systems

---

## Community & Support

- **Framework:** [Google ADK Documentation](https://ai.google.dev/adk/docs)
- **Community:** Skool group (mentioned in root README)
- **Issues:** GitHub issues for this repository
- **Models:** Latest Gemini models available on [Google AI Studio](https://aistudio.google.com)

---

## Repository Statistics

- **Total Python Files:** 77
- **Example Directories:** 12
- **Custom Projects:** 1 (auhdhd_rag_agent)
- **Documentation Files:** 15+ READMEs + GEMINI_MODELS.md
- **Docker Files:** 2 (Dockerfile, docker-compose.yml)
- **Total Size:** ~720KB
- **Last Updated:** November 16, 2025
- **Repository Status:** Clean, no uncommitted changes

---

## Quick Reference: Pattern Selection

| Use Case | Example |
|----------|---------|
| Learn ADK basics | 1-basic-agent |
| Add capabilities to agent | 2-tool-agent |
| Multi-provider LLM support | 3-litellm-agent |
| Guaranteed JSON output format | 4-structured-outputs |
| Conversation memory | 5-sessions-and-state |
| Persistent data storage | 6-persistent-storage |
| Multiple agents working together | 7-multi-agent |
| Complex team workflows | 8-stateful-multi-agent |
| Monitor agent behavior | 9-callbacks |
| Define sequential workflows | 10-sequential-agent |
| Parallel execution | 11-parallel-agent |
| Iterative refinement | 12-loop-agent |
| RAG with knowledge base | auhdhd_rag_agent |

---

*This documentation serves as a comprehensive reference for the Agent Development Kit crash course repository. For specific technical questions about any example, refer to the README.md files in individual example directories.*
