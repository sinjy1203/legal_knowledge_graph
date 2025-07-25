from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage
from search_knowledge_graph import agent
from search_knowledge_graph.state import State

load_dotenv(override=True)


async def main():
    input = State(messages=[HumanMessage(content="Consider the Acquisition Agreement between Parent \"SUPERNUS PHARMACEUTICALS, INC.\" and Target \"ADAMAS PHARMACEUTICALS, INC.\"; What is the Type of Consideration")])    
    config = {
        "configurable": {
            "max_execute_tool_count": 10
        }
    }
    result = await agent.ainvoke(input, config)
    print()

if __name__ == "__main__":
    asyncio.run(main())