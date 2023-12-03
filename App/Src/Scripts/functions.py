import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from sklearn.metrics.pairwise import cosine_distances
import warnings
warnings.filterwarnings("ignore")



def create_embeddings(texts):
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

   
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


def question_user(q:str):
  data_question = pd.DataFrame()
  emb = []
  q_list = []
  emb.append(create_embeddings(q))
  q_list.append(q)
  data_question["pregunta"] = q_list
  data_question["embedding_pregunta"] = emb
  return data_question



def data_metadata(data:object, data_p:object):
    data["distance_cos"] = data["embeddings_content"].apply(lambda x:cosine_distances(data_p.iloc[0,1],x[0].reshape(1,-1)))
    data = data.sort_values(by = "distance_cos",ascending=True)
    return data

def data_main(data:object, data_p:object):
    data = data_metadata(data, data_p)
    # data = filtered_data(param, data, "distance_cos")
    return data.iloc[0:10,:]

def documents_prompt(documents:list, pages:list):
    docs = []
    for doc in range(len(documents)):
        val_aux = documents[doc].metadata["page"]
        if val_aux in pages:
            docs.append(documents[doc])
        else:
            continue
    return docs

# Función que retorna una lista con las páginas que va a recibir el prompt
def get_pages(data:object, col_name:str):
    pages_content = []
    index_col = data.columns.get_loc(col_name)
    for index in range(data.shape[0]):
        val = data.iloc[index, index_col]
        pages_content.append(val["page"])
        pages_content = sorted(list(set(pages_content)))
    return pages_content

def function_main_documents(question:str, data:pd.DataFrame, documents:list):
    question = question_user(question)
    data_documents = data_main(data,question)
    pages = get_pages(data_documents, "metadata")
    docs_ = documents_prompt(documents, pages)
    return docs_







