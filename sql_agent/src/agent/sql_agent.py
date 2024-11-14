import os
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from agent.tools import calculator


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    is_last_step: IsLastStep


models: dict[str, BaseChatModel] = {}
if os.getenv("OPENAI_API_KEY") is not None:
    models["gpt-4o-mini"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

if not models:
    print("No LLM available. Please set API keys to enable at least one LLM.")
    if os.getenv("MODE") == "dev":
        print("FastAPI initialized failed. Please use Ctrl + C to exit uvicorn.")
    exit(1)

model = models.get("gpt-4o-mini")
db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
tools.append(calculator)

dialect = "PostgreSQL"
top_k = 5
instructions = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model_runnable = wrap_model(model)
    response = await model_runnable.ainvoke(state, config)

    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_edge(START, "model")
# Always run "model" after "tools"
agent.add_edge("tools", "model")
agent.add_conditional_edges(
    "model", pending_tool_calls, {"tools": "tools", "done": END}
)

sql_agent = agent.compile(
    checkpointer=MemorySaver(),
)
