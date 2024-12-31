from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.8)
tools = load_tools(["wikipedia", 'llm-math'], llm = llm)
 
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)
agent.run("When was Elon Musk born? What is his age right now in 2023")