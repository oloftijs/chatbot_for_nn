

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma
llm = ChatOpenAI(temperature=0.1)
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
embedding = OpenAIEmbeddings()

def load():
    pathlist = Path('.\data').glob('**/*.txt')
    loaders = [
    ]
    for path in pathlist:
        loaders.append((TextLoader(str(path), encoding='utf-8')))
    #     print(path)
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs
# Split
def Split(docs, chunkSize, chunkOverlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunkSize,
        chunk_overlap = chunkOverlap
    )

    splits = text_splitter.split_documents(docs)
    return splits


def Embed(splits):
    persist_directory = 'docs/chroma/'
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb



# ### Similarity Search
def similarity_search(vectordb,input,k=3):
    docs = vectordb.max_marginal_relevance_search(input,k=k)
    vectordb.persist()
    return docs


def full_similarity_search(vectordb,input,k=3):
    Embed(Split(load(), 1000, 100))
    docs = vectordb.max_marginal_relevance_search(input,k=k)
    vectordb.persist()
    return docs
# metadata_field_info = [
#     AttributeInfo(
#        name = "source",
#        description="The location of the source file",
#        type= "string"
#     ),
# ]

# def retreiver(query): #mochten we nog iets met source definieren willen doen
#     return SelfQueryRetriever.from_llm(
#     llm, 
#     vectordb,
#     query,
#     metadata_field_info,
#     verbose=True)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def process_file(file): #processes file names 
    file = str(file) 
    file = file[file.find(":")+9:]
    file = file.replace("'}","")
    return file
def get_prompt(input,context):
    #prompt for avoiding hallucinations as much as possible 
    context_dict= {}
    for file in context:
        context_dict[process_file(file.metadata)] = file.page_content
    return f"""
    you are an intelligent assistant.
    you only answer questions based on your given context.
    if you can not answer using the context, DO NOT answer the question
    your given context, in the format {{source: content}}
    {context_dict}
    cite your sources at the end.
    the question asked by the user
    {input}

    """

def format_sources(sources_input):
    
    sources_input = [source.metadata for source in sources_input]
    sources = []
    for source in sources_input:
        sources.append(process_file(source))
    sources = list(dict.fromkeys(sources))
    return sources


vectordb = Chroma.from_documents(
    documents=load(),
    embedding=embedding,
    persist_directory='docs/chroma/'
)

