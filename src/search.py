from dotenv import load_dotenv
import asyncio
import json
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
                # content="Consider the Acquisition Agreement between Parent \"LVMH Moët Hennessy-Louis Vuitton SE\" and Target \"Tiffany & Co.\"; Information about the Closing Condition: Compliance with Covenants"
                # content="Consider the Acquisition Agreement between Parent \"LVMH Moët Hennessy-Louis Vuitton SE\" and Target \"Tiffany & Co.\"; What is the Target's Representation & Warranty of No Material Adverse Effect, with regards to some specified date"
                # content="Consider the Merger Agreement between \"First Bancorp\" and \"Select Bancorp, Inc.\"; What is the Definition of \"Superior Proposal\""
                content="Consider the Acquisition Agreement between Parent \"Project Metal Parent, LLC\" and Target \"Medallia, Inc.\"; Information about the Closing Condition: Compliance with Covenants"
            )
        ]
    }
    context = {
        "max_execute_tool_count": 15
    }
    config = {"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": ["search"]}, "recursion_limit": 100}
    result = await agent.ainvoke(input, context=context, config=config)
    response_tool_result = []
    for message in result['messages']:
        if message.type == "tool" and message.name == "ResponseTool":
            response_tool_result.extend(json.loads(message.content))

    retrieved_snippets = []
    for i, chunk_info in enumerate(response_tool_result):
        retrieved_snippets.append(
            {
                "file_path": chunk_info['file_path'],
                "span": chunk_info['span'],
                "score": 1.0 / (i + 1)
            }
        )

    print(retrieved_snippets)


if __name__ == "__main__":
    asyncio.run(main())
