import csv
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document 
from transformers import CanineModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


os.environ["HF_HOME"] = "./cache/"
df= pd.read_csv('./IM_RAG_Ans_data.csv')


#load and split
columns_to_embed = df.columns.tolist()
columns_to_metadata = df.columns.tolist()
dir_path = './IM_RAG_Ans_data.csv'

docs = []
with open(dir_path, newline="") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, 
                                chunk_overlap=100,
                                length_function=len)
documents = splitter.split_documents(docs)

#DB 구축
results = {}
model_name='google/canine-c'
model = CanineModel.from_pretrained(model_name)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

#Save Disk
vectorstore = Chroma.from_documents(documents,
                                hf,
                                persist_directory= f"./documents_test_db",
                                collection_metadata = {'hnsw:space': 'cosine','dimension': model.config.hidden_size})

vectorstore.persist()
vectorstore = None

vectorstore=Chroma(persist_directory=f"./documents_test_db", embedding_function=hf)