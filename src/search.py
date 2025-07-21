import os
from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage
from search_knowledge_graph.state import State, Config
from search_knowledge_graph.agent import ReactAgent
from search_knowledge_graph.tools import KnowledgeGraphSearchTool

load_dotenv(override=True)


knowledge_graph_search_tool = KnowledgeGraphSearchTool()

agent = ReactAgent(
    model_kwargs={
        "base_url": os.getenv("LLM_BASE_URL"),
        "model": os.getenv("LLM_MODEL"),
        "temperature": 0.1,
        "api_key": "dummy"
    },
    tools=[
        knowledge_graph_search_tool
    ],
    graph_schema=knowledge_graph_search_tool.get_graph_schema()
)


async def main():
    input = State(messages=[HumanMessage(content="Consider the Acquisition Agreement between Parent \"SUPERNUS PHARMACEUTICALS, INC.\" and Target \"ADAMAS PHARMACEUTICALS, INC.\"; What is the Type of Consideration")])    
    config = {
        "configurable": {
            "max_execute_tool_count": 5
        }
    }
    _ = await agent.ainvoke(input, config)

if __name__ == "__main__":
    asyncio.run(main())