from langchain_core.messages import HumanMessage # high level fw to build ai apps
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent # allows to build ai agents 
from dotenv import load_dotenv # allows loading env variables in python script 

load_dotenv() # loads in vars
# agent vs bot: agent has access to tools 

@tool 
def calculator(a: float, b: float) -> str:
    # doc string: desc of function so llm knows when to use it 
    """Useful for performing basic arithmetic calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

@tool 
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"

def main():
    # use an llm for the agent's "brain"
    model = ChatOpenAI(temperature=0) # temp = 0 = no randomness
    
    tools = [calculator, say_hello]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit")
    print("You can ask me to perform calculations or chat with me :)")

    # so that user can keep asking questions continuously (unless quit)
    while True: 
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break 

        print("\nAssistant: ", end="")
        # loop through all of chunks (part of the agent's response) check to see if there is agent response 
        for chunk in agent_executor.stream(
            # stream response from llm with agent exec
            # have to call agent to use it
            {"messages": [HumanMessage(content=user_input)]}
        ):
            # are there messages in the agent response? 
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    # stream longer response from agent, so it looks like agent is trying word by word,
                    #  not all at once 
                    print(message.content, end="")
        print()
    
# execute main fuction IF we execute this file directly 
if __name__ == "__main__":
    main()
        


