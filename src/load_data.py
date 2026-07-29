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
from langchain_community.document_loaders import JSONLoader
import os
from typing import List, Any
from pathlib import Path
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

def load_data():
    data = []

    for single_file in json_paths:
        try:
            with open(single_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data.append(json.loads(line))

            print(f"✓ Loaded {single_file} - {len(data)} objects so far")

        except json.JSONDecodeError as e:
            print(f"JSON error in {single_file}, line {line_num}: {e}")
        except Exception as e:
            print(f"✗ Error loading {single_file}: {e}")

    return data


if __name__ == "__main__":
    data = data 