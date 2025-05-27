import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .retrieve_knowledge_tool import query_knowledge_base

# Load environment variables from .env file
load_dotenv()

# 1. Configure LiteLLM
# Get the API key from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
    print("Please ensure it is set in your .env file or environment.")
    # Optionally, exit or raise an error if the key is critical for operation
    # exit() # or raise ValueError("OPENROUTER_API_KEY not set")

llm_model = LiteLlm(
    model_name="openrouter/anthropic/claude-3-haiku-20240307", # Or your preferred model
    api_key=openrouter_api_key
)

# 2. Define the ADK Agent
# System prompt/instruction for the LLM
agent_instruction = """
You are an AI assistant designed to be helpful and understanding, especially for users who may be AuDHD.
Your primary goal is to answer questions based on information retrieved from a knowledge base.

Guidelines for your responses:
- Be clear, concise, and direct.
- Break down complex information into smaller, easy-to-understand parts.
- If asked, provide information step-by-step.
- Use simple language and avoid jargon where possible. If technical terms are necessary, briefly explain them.
- Be patient and literal in interpreting questions.

How to answer:
1. ALWAYS prioritize using the 'query_knowledge_base' tool to search for relevant information before answering, especially if the user's query seems to seek specific knowledge.
2. If the tool returns relevant information, synthesize your answer based *only* on that information and the user's query. Clearly indicate that the information comes from the knowledge base.
3. If the tool returns no relevant information or an error, politely inform the user that you couldn't find specific information in the knowledge base on that topic. Do not try to answer from general knowledge if the query seemed to imply it should be in the knowledge base.
4. If the query is a general greeting, request for clarification, or a simple conversational interaction that does not require the knowledge base, you may respond directly.
"""

rag_agent = Agent(
    name="AuDHDRAGAgent",
    model=llm_model,
    tools=[query_knowledge_base],
    description="An AuDHD-friendly RAG agent that answers questions based on a knowledge base.",
    instruction=agent_instruction
)

if __name__ == '__main__':
    print("AuDHD RAG Agent initialized.")
    print(f"Agent Name: {rag_agent.name}")
    print(f"Agent Description: {rag_agent.description}")
    print(f"Using Model: {llm_model.model_name}")
    print(f"Tools available: {[tool.__name__ for tool in rag_agent.tools]}")

    # Example of how one might interact with the agent (actual interaction loop not implemented here)
    # This is for demonstration; direct invocation might differ based on ADK usage.
    # print("\n--- Example Interaction (Conceptual) ---")
    # test_query = "What is a RAG model?"
    # print(f"User Query: {test_query}")
    #
    # # In a real scenario, the ADK framework would handle the agent's execution flow.
    # # This is a simplified representation of how the agent might process a query:
    #
    # # 1. Agent receives query.
    # # 2. Agent decides to use query_knowledge_base tool.
    # # context_result = query_knowledge_base(test_query)
    # #
    # # # 3. Agent uses LLM with context to generate response.
    # # if context_result.get("status") == "success" and context_result.get("retrieved_context"):
    # #     prompt_for_llm = f"{agent_instruction}\n\nKnowledge Base Information:\n{context_result['retrieved_context']}\n\nUser Query: {test_query}\n\nAnswer:"
    # #     # response = llm_model.generate_text(prompt_for_llm) # Simplified
    # #     # print(f"Agent Response (hypothetical): {response}")
    # #     print("Agent would use the retrieved context to answer.")
    # # elif context_result.get("status") == "error":
    # #     print(f"Agent Response (hypothetical): I encountered an error trying to access my knowledge base: {context_result.get('message')}")
    # # else:
    # #     print("Agent Response (hypothetical): I couldn't find specific information in my knowledge base for your query.")
    #
    # print("\nNote: To run a full interaction, integrate this agent into an ADK application or use ADK's testing utilities.")
