from langchain_classic.agents import Tool

def calculator(query: str) -> str:
    try:
        return eval(query)
    except Exception as e:
        return "Error: " + str(e)

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="A simple calculator to evaluate mathematical expressions."
)
