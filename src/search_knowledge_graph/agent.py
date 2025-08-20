import json
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from .state import State, ContextSchema
from .prompt import SYSTEM_TEMPLATE


class ReactAgent:
    def __new__(cls, model_kwargs, tools):
        instance = super().__new__(cls)
        instance.__init__(model_kwargs, tools)
        return instance.graph

    def __init__(self, model_kwargs, tools):
        llm = ChatOpenAI(**model_kwargs)
        self.system_prompt = SystemMessage(content=SYSTEM_TEMPLATE)
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}

        workflow = StateGraph(State, context_schema=ContextSchema)
        workflow.add_node("llm", self.llm)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("end", self.end)

        workflow.add_edge(START, "llm")
        self.graph = workflow.compile()

    async def llm(self, state, runtime: Runtime[ContextSchema]):
        response = await self.llm_with_tools.ainvoke([self.system_prompt] + state.messages)

        if response.tool_calls:
            if (
                state.execute_tool_count
                >= runtime.context.max_execute_tool_count
            ):
                update = {"messages": [AIMessage(content="도구 실행 횟수를 초과했습니다.")]}
                goto = "end"
            else:
                update = {"messages": [response]}
                goto = "execute_tool"
        else:
            update = {"messages": [AIMessage(content=response.content)]}
            goto = "end"

        return Command(update=update, goto=goto)

    async def execute_tool(self, state, runtime: Runtime[ContextSchema]):
        outputs = []

        tasks = []
        for tool_call in state.messages[-1].tool_calls:
            task = self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            tasks.append((tool_call, task))

        results = await asyncio.gather(*[task for _, task in tasks])

        for (tool_call, _), result in zip(tasks, results):
            outputs.append(
                ToolMessage(
                    name=tool_call["name"],
                    content=json.dumps(result, indent=2, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )

        update = {
            "messages": outputs,
            "execute_tool_count": state.execute_tool_count + 1,
        }
        if outputs[-1].name == "ResponseTool":
            goto = "end"
        else:
            goto = "llm"

        return Command(update=update, goto=goto) 
    
    async def end(self, state, runtime: Runtime[ContextSchema]):
        if runtime.context.progress_bar:
            runtime.context.progress_bar.update(1)
        return Command(goto=END)