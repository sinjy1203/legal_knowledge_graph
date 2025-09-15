import random
from typing_extensions import Self
from collections.abc import Sequence
from pydantic import BaseModel, model_validator, computed_field
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime
from generate_knowledge_graph.utils.model import Document
from generate_knowledge_graph.state import ContextSchema


MAX_TESTS_PER_BENCHMARK = 194

logger = setup_logger()

class Snippet(BaseModel):
    file_path: str
    span: tuple[int, int]

    @computed_field  # type: ignore[misc]
    @property
    def answer(self) -> str:
        with open(f"./data/corpus/{self.file_path}") as f:
            return f.read()[self.span[0] : self.span[1]]

def validate_snippet_list(snippets: Sequence[Snippet]) -> None:
    snippets_by_file_path: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if snippet.file_path not in snippets_by_file_path:
            snippets_by_file_path[snippet.file_path] = [snippet]
        else:
            snippets_by_file_path[snippet.file_path].append(snippet)

    for _file_path, snippets in snippets_by_file_path.items():
        snippets = sorted(snippets, key=lambda x: x.span[0])
        for i in range(1, len(snippets)):
            if snippets[i - 1].span[1] >= snippets[i].span[0]:
                raise ValueError(
                    f"Spans are not disjoint! {snippets[i - 1].span} VS {snippets[i].span}"
                )


class QAGroundTruth(BaseModel):
    query: str
    snippets: list[Snippet]
    tags: list[str] = []

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        validate_snippet_list(self.snippets)
        return self


class Benchmark(BaseModel):
    tests: list[QAGroundTruth]


class DataLoader:
    def __call__(self, state, runtime: Runtime[ContextSchema]):
        logger.info("Loading data...")
        
        all_tests: list[QAGroundTruth] = []
        document_file_paths_set: set[str] = set()
        used_document_file_paths_set: set[str] = set()
        with open(f"./data/benchmarks/{runtime.context.benchmark_name}.json", encoding="utf-8") as f:
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
                test.tags = [runtime.context.benchmark_name]
            all_tests.extend(tests)
        

        benchmark = Benchmark(
            tests=all_tests,
        )

        # Create corpus (sorted for consistent processing)
        corpus: list[Document] = []
        for document_file_path in sorted(used_document_file_paths_set):
            # if document_file_path != "maud/DSP_Group_Synaptics_Incorporated.txt" and document_file_path != "maud/Adamas_Pharmaceuticals_Supernus_Pharmaceuticals.txt":
            # if document_file_path != "maud/TIFFANY_&_CO._LVMH_MOËT_HENNESSY-LOUIS_VUITTON.txt":
            #     continue
            with open(f"./data/corpus/{document_file_path}", encoding="utf-8") as f:
                corpus.append(
                    Document(
                        file_path=document_file_path,
                        content=f.read(),
                    )
                )
        
        logger.info(f"loaded {len(corpus)} documents")
        
        return Command(
            update={
                "documents": corpus
            },
            goto="IntroBodySeparator"
        )