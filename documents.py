from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveJsonSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)

# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
import os 

load_dotenv()

import json
data = []


# Load json Data

# Json data paths.
json_paths = ["data/convfinqa_corpus.jsonl/corpus.jsonl",
              "data/financebench_corpus.jsonl/corpus.jsonl",
              "data/finder_corpus.jsonl/corpus.jsonl",
              "data/finqa_corpus.jsonl/corpus.jsonl",
              "data/finqabench_corpus.jsonl/corpus.jsonl",
              "data/multiheirtt_corpus.jsonl/corpus.jsonl",
              "data/tatqa_corpus.jsonl/corpus.jsonl"]


for single_file in json_paths:
  with open(single_file, 'r') as f:
    try:
        with open(single_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    json_obj = json.loads(line)
                    data.append(json_obj)
        
        print(f'✓ Loaded {single_file} - {len(data)} objects so far')
    
    except json.JSONDecodeError as e:
        print(f'JSON error in {single_file}, line {line_num}: {e}')
    except Exception as e:
        print(f'✗ Error loading {single_file}: {e}')

print(f'\nTotal documents loaded: {len(data)}')



# Semantic chunking (json)
splitter = RecursiveJsonSplitter(
    max_chunk_size=1000,
    min_chunk_size = 200)

chunks = splitter.split_text(data)