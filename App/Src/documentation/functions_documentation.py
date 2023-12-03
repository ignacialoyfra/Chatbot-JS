
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import warnings
warnings.filterwarnings("ignore")



def create_embeddings(texts):
    # Cargar el modelo y el tokenizador
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenizar los textos y convertirlos a tensores
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Obtener los embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convertir los embeddings a una matriz numpy
    embeddings = embeddings.numpy()

    return embeddings


def function_text_splitter(document, chunk_size:int, chunk_overlap:int):
        data = document.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        length_function = len,
        chunk_overlap = chunk_overlap
        )
        documents_result = text_splitter.split_documents(data)
        return documents_result


def dataframe_documentation(documents:list):
    data = pd.DataFrame()
    page_content = []
    metadata = []
    for line in range(len(documents)):
        page_content.append(documents[line].page_content)
        metadata.append(documents[line].metadata)
    data["page_content"] = page_content
    data["metadata"] = metadata
    return data

