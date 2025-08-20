from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from search_knowledge_graph import agent
from search_knowledge_graph.state import State

load_dotenv(override=True)


async def main():
    langfuse_handler = CallbackHandler()
    
    input = {
        "messages": [
            HumanMessage(
                content="Consider the Acquisition Agreement between Parent \"LVMH Moët Hennessy-Louis Vuitton SE\" and Target \"Tiffany & Co.\"; Information about the Closing Condition: Compliance with Covenants"
            )
        ]
    }
    context = {
        "max_execute_tool_count": 10
    }
    config = {"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": ["search"]}}
    result = await agent.ainvoke(input, context=context, config=config)
    print()

if __name__ == "__main__":
    asyncio.run(main())