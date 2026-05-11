import os  
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.agents import Tool, create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.prompts import PromptTemplate

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def calculator(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return "Error: " + str(e)

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="A simple calculator to evaluate mathematical expressions."
)

def get_react_agent():
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator_tool]
    
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
