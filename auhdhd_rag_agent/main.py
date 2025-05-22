import os
from dotenv import load_dotenv
from google.adk.runtime import Runner
from .agent import rag_agent # Assuming rag_agent is an instance of Agent

def run_cli():
    """Runs the Command Line Interface for the AuDHD RAG Agent."""

    # Load environment variables from .env file
    # This is crucial for the agent to access API keys, etc.
    load_dotenv()

    # Initialize the ADK Runner
    # For now, no session_service is provided for simplicity in a basic CLI.
    # A session service would be needed for conversation history.
    try:
        runner = Runner(
            agent=rag_agent,
            app_name="AuDHD_RAG_CLI"
            # session_service=... # Can be added later if history is needed
        )
        print("Runner initialized successfully.")
    except Exception as e:
        print(f"Error initializing ADK Runner: {e}")
        print("Please ensure your agent.py is correctly defined and OPENROUTER_API_KEY is set.")
        return

    # CLI Interaction Loop
    print("\nAuDHD RAG Agent CLI. Type 'exit' to quit.")
    print("Note: Ensure you have run 'python auhdhd_rag_agent/ingest.py' to build the knowledge base.")
    print("      And that your OPENROUTER_API_KEY is correctly set in .env")


    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting CLI. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            print("Sending query to agent...")
            # The runner.run() method processes the query through the agent
            # and its tools, then gets a response from the LLM.
            response_data = runner.run(query=user_input)

            # The response_data is typically a dictionary.
            # The actual structure might vary slightly based on ADK version or
            # if a custom response model is defined for the agent.
            # For a simple agent response, it's often in response_data['response'].
            agent_response = response_data.get('response', 'No response content found.')
            print(f"Agent: {agent_response}")

        except Exception as e:
            print(f"An error occurred while processing your query: {e}")
            # Optionally, print more detailed error for debugging
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    run_cli()
