import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Uncomment if you want to temporarily disable logger
logging.disable(sys.maxsize)

# # NOTE: only necessary for querying with `use_async=True` in notebook
# import nest_asyncio

# nest_asyncio.apply()

# My OpenAI Key
import os
# from openai import OpenAI
os.environ["OPENAI_API_KEY"] = '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'
from llama_index import (
    TreeIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    VectorStoreIndex,
    SummaryIndex,
    PromptTemplate,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.base import BaseIndex
from llama_index.llms.base import LLM
from llama_index.llms import OpenAI
from llama_index.response.schema import Response
import pandas as pd
from typing import Tuple

from pathlib import Path

import requests
from dataclasses import dataclass
from typing import List
import re
documents = SimpleDirectoryReader("data").load_data()

def load():

    if not os.path.exists("./storage"):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist()
    else:
    # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    return index
