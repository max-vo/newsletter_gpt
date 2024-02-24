import pandas as pd
import openai_api_functions as ai
from pathlib import Path
import numpy as np

def create_embeddings_from_df(df:pd.DataFrame):
    keywords = df["keyword"].tolist()
    embeddings = [None]*len(keywords)
    client = ai.create_client()
    for i in range(len(keywords)):
        embeddings[i] = ai.create_embedding(keywords[i], client)

    keyword_embedding_dataframe = pd.DataFrame(keywords, embeddings, columns=["keywords"]+list(range(len(embeddings[0])+1)))
    return keyword_embedding_dataframe

def create_vector_database(filename="database_pre.csv"):
    df = pd.read_csv(filename)
    key_embeddings = create_embeddings_from_df(df)
    key_embeddings.to_csv("database.csv", index=True)
    return key_embeddings
def load_database(filename="database.csv"):
    db = pd.read_csv(filename)
    # vectorize

    return db

def search_for_key(query):
    embedding = ai.create_embedding(query)
    return key