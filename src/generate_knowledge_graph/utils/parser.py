import json
import re
from langchain_core.output_parsers import StrOutputParser


class JsonOutputParser(StrOutputParser):
    def __init__(self):
        super().__init__()
    def parse(self, llm_output: str):
        text = super().parse(llm_output)
        
        # </think> 이전의 모든 문자열 제거
        think_pattern = r'.*</think>'
        text = re.sub(think_pattern, '', text, flags=re.DOTALL)
        
        pattern = r'```json\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
