import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English


#get documents location.
pdf_path = "Example2.pdf"


#data formatting cuz its needed for vector databases. here i used basic division based on spaces and sentences based on full stops.
def text_formatter(text: str) -> str:
    """Performs minor formatting on text"""
    cleaned_text = text.replace("\n", " ").strip()

    #more formatting if needed.
    return cleaned_text


#function to read pdf. ive used fitz to read the pdf file. this is a library that allows this and i dont still have enough knowledge to also do this on pure python with my own functions.
def open_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_text.append({"page_number": page_number,
                           "page_char_count": len(text),
                           "page_word_count": len(text.split(" ")),
                           "page_sentence_count": len(text.split(". ")),
                           "page_token_count": len(text)/4,
                           "text": text})
    return pages_text

pagestexts = open_read_pdf(pdf_path=pdf_path)


#bringing all data into a dataframe as pandas can work with it easily.
# df = pd.DataFrame(pagestexts)
# print(df.head())
# print(df.describe().round(2))


# now for the formatting part formatting the pdf text to be then divided into tokens. these limits might reduce the quality and hence we divide it already. we using spacy here
nlp = English()

nlp.add_pipe("sentencizer")

doc = nlp("this is a sentence. this is another sentence. i like cheetahs.")
assert len(list(doc.sents)) == 3
# print(list(doc.sents))

for item in tqdm(pagestexts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    item["page_sentence_count_spacy"] = len(item["sentences"])

# df = pd.DataFrame(pagestexts)
# print(df.head())
# print(df.describe().round(2))


# chunking our sentences together. no 100% correct way, just experiment and then use what siuts best
# can be done using textsplitters from langchain however here its done with pure python.
num_sentence_chunk_size = 10

def split_list(input_list: list[str], slice_size: int=num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

# test_list = list(range(25))
# print(split_list(test_list))

for item in tqdm(pagestexts):
    item["sentence_chunks"] = split_list(input_list= item["sentences"], slice_size= num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

# df = pd.DataFrame(pagestexts)
# print(df.head())
# print(df.describe().round(2))


# splitting each chunk into its own item.embed each chunk into its own numerical representation. this will provide us with a good level of granularity. this means we can specifically use each parts as a sample.
import re
pageschunks = []
for item in tqdm(pagestexts):
    for sentencechunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        joined_sentence_chunk = "".join(sentencechunk).replace(" ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

        chunk_dict["sentencechunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk)/4

        pageschunks.append(chunk_dict)

# print(len(pageschunks))
# print(random.sample(pageschunks, k=1))
df = pd.DataFrame(pageschunks)
# print(df.head())
# print(df.describe().round(2))


# limiting the minimum token size for chunks.
# remove chunks with tokens lower than 20 tokens
min_token_length = 20
# for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
#     print(f'chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentencechunk"]}')

pageschunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient = "records")

# print(random.sample(pageschunks_over_min_token_len, k=1))


# embedding our text chunks into database, or what i call it as the miracle space.
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="BAAI/bge-small-en-v1.5", device="cuda")

for item in tqdm(pageschunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentencechunk"])

text_chunks = [item["sentencechunk"] for item in pageschunks_over_min_token_len]
text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)

# print(pageschunks_over_min_token_len[56])
text_chunks_and_embeddings_df = pd.DataFrame(pageschunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df2.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index = False)

text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load.head()


import random
import torch
import numpy as np 
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)

# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
# embeddings.shape

from sentence_transformers import util

embedding_model = SentenceTransformer(model_name_or_path="BAAI/bge-small-en-v1.5",  device=device) # choose the device to load the model to


# 1. Define the query
# Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
query = "Structured Data"
print(f"Query: {query}")

# 2. Embed the query to the same numerical space as the text examples 
# Note: It's important to embed your query with the same model you embedded your examples with.
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# 3. Get similarity scores with the dot product (we'll time this for fun)
from time import perf_counter as timer

start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

# 4. Get the top-k results (we'll keep this to 5)
top_results_dot_product = torch.topk(dot_scores, k=5)
top_results_dot_product

import textwrap
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


print(f"Query: '{query}'\n")
print("Results:")
# Loop through zipped together scores and indicies from torch.topk
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentencechunk"])
    # Print the page number too so we can reference the textbook further (and check the results)
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

