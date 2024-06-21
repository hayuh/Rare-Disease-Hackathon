import os
import sys
from helper import get_openai_api_key

venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'python3.12', 'site-packages')
sys.path.append(venv_path)

os.environ["OPENAI_API_KEY"] = get_openai_api_key()

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("Ehlers-Danlos-1")
documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source']

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(testset)
testset.to_pandas()