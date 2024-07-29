#from langchain.document_loaders.csv_loader import CSVLoader

#from langchain.llms import CTransformers
import sys
import os
import sentence_transformers
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings#, OllamaEmbeddings
from langchain_community.vectorstores import FAISS, chroma

#DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path="stocklive2.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

#db = chroma.from_documents(text_chunks[:20],HuggingFaceEmbeddings())







# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings()#model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks[:20], embeddings)

#docsearch.save_local(DB_FAISS_PATH)


#query = "What is the value of GDP per capita of Finland provided in the data?"

#docs = docsearch.similarity_search(query, k=3)

query = "What is the LTP of PRIN symbol"

docs = docsearch.similarity_search(query, k=3)

sec_key = os.getenv("huggtoken1")

os.environ["huggtoken1"]= sec_key

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",#"mistralai/Mistral-7B-v0.1","gpt2","meta-llama/Meta-Llama-3-8B"
    task="text-generation",
    #tokenizer = AutoTokenizer.from_pretrained("gpt2"),
    #model = AutoModelForCausalLM.from_pretrained("gpt2"),
    model_kwargs={"temperature": 0.8, "top_k": 50},#---->newly added
    pipeline_kwargs={ "max_new_tokens": 100}#"temperature": 0,
)

local_llm = hf
qa = ConversationalRetrievalChain.from_llm(local_llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    #query = "WWhat is the LTP of PRIN symbol?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa.invoke({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])