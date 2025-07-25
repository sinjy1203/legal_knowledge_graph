import os
import random
import json
from tqdm.auto import tqdm
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from search_knowledge_graph import agent
from search_knowledge_graph.state import State

from legalbenchrag.legalbenchrag.benchmark_types import (
    QueryResponse,
    RetrievedSnippet,
    QAGroundTruth,
    Benchmark,
)
from legalbenchrag.legalbenchrag.run_benchmark import QAResult, BenchmarkResult


MAX_TESTS_PER_BENCHMARK = 194
BENCHMARK_NAME = "maud"
BENCHMARK_RESULT_DIR = "./data/benchmark_results"


async def pred(benchmark):
    progress_bar = tqdm(total=len(benchmark.tests), desc="searching...")
    langfuse_handler = CallbackHandler()

    states = []
    configs = []
    for test_data in benchmark.tests:
        states.append(State(messages=[HumanMessage(content=test_data.query)]))
        configs.append(
            {
                "max_concurrency": 8,
                "callbacks": [langfuse_handler],
                "metadata": {
                    "benchmark": BENCHMARK_NAME,
                    "query": test_data.query,
                    "snippets": [snippet.model_dump() for snippet in test_data.snippets],
                },
                "configurable": {
                    "max_execute_tool_count": 10,
                    "progress_bar": progress_bar,
                }
            }
        )
    
    responses = await agent.abatch(states, configs)
    progress_bar.close()

    qa_results = []
    for test_data, response in zip(benchmark.tests, responses):
        retrieved_snippets = []
        if response['messages'][-1].type == "ai":
            pass
        else:
            for i, chunk_info in enumerate(json.loads(response['messages'][-1].content)):
                retrieved_snippets.append(
                    {
                        "file_path": chunk_info['file_path'],
                        "span": chunk_info['span'],
                        "score": 1.0 / (i + 1)
                    }
                )

        qa_results.append(
            QAResult(
                qa_gt=test_data.model_dump(),
                retrieved_snippets=retrieved_snippets,
            )
        )
    return BenchmarkResult(qa_result_list=qa_results, weights=[1.0] * len(responses))


def load_data():
    all_tests: list[QAGroundTruth] = []
    document_file_paths_set: set[str] = set()
    used_document_file_paths_set: set[str] = set()
    with open(f"./data/benchmarks/{BENCHMARK_NAME}.json", encoding="utf-8") as f:
        benchmark = Benchmark.model_validate_json(f.read())
        tests = benchmark.tests
        document_file_paths_set |= {
            snippet.file_path for test in tests for snippet in test.snippets
        }
        if len(tests) > MAX_TESTS_PER_BENCHMARK:
            tests = sorted(
                tests,
                key=lambda test: (
                    random.seed(test.snippets[0].file_path),
                    random.random(),
                )[1],
            )
            tests = tests[:MAX_TESTS_PER_BENCHMARK]
        used_document_file_paths_set |= {
            snippet.file_path for test in tests for snippet in test.snippets
        }
        for test in tests:
            test.tags = [BENCHMARK_NAME]
        all_tests.extend(tests)

    return Benchmark(
        tests=all_tests,
    )



async def main():
    benchmark = load_data()
    benchmark_result = await pred(benchmark)

    result_path = f"{BENCHMARK_RESULT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "results.json"), "w", encoding="utf-8") as f:
        f.write(benchmark_result.model_dump_json(indent=4))
    
    summary = {
        "average_precision": benchmark_result.avg_precision,
        "average_recall": benchmark_result.avg_recall,
    }
    with open(os.path.join(result_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())