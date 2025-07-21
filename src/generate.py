import os
from dotenv import load_dotenv
from generate_knowledge_graph.builder import graph
from generate_knowledge_graph.state import State, Config

load_dotenv(override=True)


def main():
    input = State(benchmark_name="maud", clear_database=True)    
    _ = graph.invoke(input)

if __name__ == "__main__":
    main()