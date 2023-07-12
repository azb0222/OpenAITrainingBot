from llama_index import (
    download_loader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage,
    ListIndex, LLMPredictor, load_graph_from_storage,
    ComposableGraph,
)
from pathlib import Path
import asyncio 
from langchain import OpenAI

#langchain to help setup outer chatbot agent 
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform

import webcrawler
import openai
import os 
openai.api_key = os.getenv("OPENAI_API_KEY")


from fastapi import FastAPI, HTTPException
from uuid import uuid4
from pydantic import BaseModel
from typing import Dict, List, Optional

import json 

data = [] #fix so erases the data upon start 
json_file_path = "data.json"

async def setup(): 
  loader = webcrawler.KnowledgeBaseWebReader(
        root_url='https://dev.blues.io/notecard',
        link_selectors=['a.menuHeader',],
        body_selector='div.main-container'
    )
  documents, categories = await loader.load_data()
  
  #write documents to JSON file 
  with open(json_file_path, "w") as file:
    json.dump(documents, file)

  
  index_set = await createIndicies(categories, documents)
  llm_predictor, graph = await createGraph(index_set, categories)
  setupChatBot(graph, index_set, categories, llm_predictor)
  
  

#set up a vector index for each category
#TODO: need to write load index from disk
async def createIndicies(categories, documents): 
  print("Creating indices")
  service_context = ServiceContext.from_defaults(chunk_size=512)
  index_set={}
  for category in categories: 
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        documents[category],   
        service_context=service_context,
        storage_context=storage_context)
    index_set[category] = cur_index
    storage_context.persist(persist_dir=f'./storage/{category}')
    
  return index_set 
  
#graph will help synthesize data across all vector indexes
async def createGraph(index_set, categories): 
  print("Creating graph")
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
  storage_context = StorageContext.from_defaults()
  index_summaries = [f"{category}" for category in categories]
  
  graph = ComposableGraph.from_indices(
    ListIndex,
    [index_set[category] for category in categories], 
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context = storage_context,
  )
  root_id = graph.root_id
  
  #save to disk TODO: fix to only load from disk when needed as in og api 
  storage_context.persist(persist_dir=f'./storage/root')
  
  # [optional] load from disk, so you don't need to build graph from scratch
  graph = load_graph_from_storage(
    root_id=root_id, 
    service_context=service_context,
    storage_context=storage_context,
  )
  return (llm_predictor, graph)


#TODO: pass in category
#SETUP OF THE CHAT BOT 
def setupChatBot(graph, index_set, categories, llm_predictor):
  print("Setting up chatbot") 
  #DecomeQueryTransform to each for each vector index within graph 
  decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
  )

# define custom retrievers

  custom_query_engines = {}
  for index in index_set.values():
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_metadata={'index_summary': index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
  custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
  )

  # construct query engine
  graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

# tool config
  graph_config = IndexToolConfig(
    query_engine=graph_query_engine,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    tool_kwargs={"return_direct": True}
  )

# define toolkit
  index_configs = []
  for category in categories:
    query_engine = index_set[category].as_query_engine(
        similarity_top_k=3,
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name=f"Vector Index {category}",
        description=f"useful for when you want to answer queries about a specific {category}",
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)
    
  toolkit = LlamaToolkit(
    index_configs=index_configs + [graph_config],
  )

  memory = ConversationBufferMemory(memory_key="chat_history")
  llm=OpenAI(temperature=0)
  agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
  )
  while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f'Agent: {response}')
    

asyncio.run(setup())



#API Setup: 

app = FastAPI()

conversations: Dict[str, List[str]] = {}

class Query(BaseModel):
    prompt: str

class Response(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    global final_toolkit
    final_toolkit = await setup()

@app.post("/query/{uuid}", response_model=Response)
async def query(uuid: str, query: Query):
    if uuid not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversations[uuid]
    llm = OpenAI(temperature=0)
    agent_chain = create_llama_chat_agent(
        final_toolkit,
        llm,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=True
    )
    response = agent_chain.run(input=query.prompt)
    conversation.append(f'User: {query.prompt}')
    conversation.append(f'Agent: {response}')
    
    return Response(response=response)

@app.get("/conversations/{uuid}", response_model=List[str])
async def get_conversation(uuid: str):
    if uuid not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation = conversations[uuid]
    return conversation

@app.post("/conversations/", response_model=str)
async def create_conversation():
    uuid = str(uuid4())
    conversations[uuid] = []
    return uuid
  
  
#with vectorDatabase: 

async def createIndicies(categories, documents): 
  print("Creating indices")
  service_context = ServiceContext.from_defaults(chunk_size=512)
  index_set = {}
  for category in categories: 
    # Transform documents to vectors and store them in your vector database
    vector_data = some_transformation_function(documents[category])
    cur_index = MyVectorDatabase(vector_data)
    
    # Save your vector database to disk
    cur_index.save_to_disk(f'./storage/{category}')
    
    index_set[category] = cur_index
  return index_set