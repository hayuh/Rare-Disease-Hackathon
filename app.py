import os
import glob
from pathlib import Path
import gradio as gr
import nest_asyncio

# Ensure async compatibility in Jupyter
nest_asyncio.apply()

# Import OpenAI key with helper function
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

# Define the path to the directory containing the PDF files
folder_path = 'Ehlers-Danlos-1'

# Get the list of all PDF files in the directory
pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
print(pdf_files)

# Extract just the filenames (optional)
pdf_filenames = [os.path.basename(pdf) for pdf in pdf_files]
print(pdf_filenames)

# Import utilities
from utils import get_doc_tools

# Truncate function names if necessary
def truncate_function_name(name, max_length=64):
    return name if len(name) <= max_length else name[:max_length]

# Create tools for each PDF
paper_to_tools_dict = {}
for pdf in pdf_files:
    print(f"Getting tools for paper: {pdf}")
    vector_tool, summary_tool = get_doc_tools(pdf, Path(pdf).stem)
    paper_to_tools_dict[pdf] = [vector_tool, summary_tool]

# Combine all tools into a single list
all_tools = [t for pdf in pdf_files for t in paper_to_tools_dict[pdf]]

# Define an object index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# Initialize the OpenAI LLM
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

# Set up the agent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Define the function to query the agent
def ask_agent(question):
    response = agent.query(question)
    return str(response)

# Create the Gradio interface
iface = gr.Interface(
    fn=ask_agent,
    inputs="text",
    outputs="text",
    title="EDS Diagnosis Helper",
    description="Ask questions related to Ehlers-Danlos Syndrome diagnosis.",
)

# Launch the Gradio app
iface.launch(share=True)

"""
import streamlit as st
from transformers import pipeline

# Load your model
generator = pipeline('text-generation', model='gpt-3.5-turbo')

# Streamlit interface
st.title("Text Generator")
prompt = st.text_input("Enter your prompt:")
if st.button("Generate"):
    result = generator(prompt, max_length=50)
    st.write(result[0]['generated_text'])

"""
    
"""
import gradio as gr
from transformers import pipeline

# Load your model
generator = pipeline('text-generation', model='gpt-3.5-turbo')

# Define the function to generate text
def generate_text(prompt):
    result = generator(prompt, max_length=50)
    return result[0]['generated_text']

# Create the Gradio interface
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Text Generator")

# Launch the interface
iface.launch()
"""
"""
import torch
print(torch.__version__)
"""