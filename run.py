import torch
import numpy as np 
import pandas as pd


embeddings_df_save_path = "text_chunks_and_embeddings_df2.csv"
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load.head()


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

from sentence_transformers import util, SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="BAAI/bge-small-en-v1.5",  device=device) # choose the device to load the model to


# 1. Define the query
# Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
query = "Boolean Algebra"

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


# print(f"Query: '{query}'\n")
# print("Results:")
# # Loop through zipped together scores and indicies from torch.topk
# for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
#     print(f"Score: {score:.4f}")
#     # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
#     print("Text:")
#     print_wrapped(pages_and_chunks[idx]["sentencechunk"])
#     # Print the page number too so we can reference the textbook further (and check the results)
#     print(f"Page number: {pages_and_chunks[idx]['page_number']+1}")
#     print("\n")


embeddings_cpu = embeddings.to("cpu")
import numpy as np
embeddings_cpu = embeddings_cpu / np.linalg.norm(embeddings_cpu, axis=1, keepdims=True)
embeddings = embeddings_cpu.to("cuda")

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    # if print_time:
    #     print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentencechunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")



import torch

# Load model and tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = './BART_model'  # Path to the folder containing the model files
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm_model.to("cuda")

# print(llm_model)
# okay, so the model we're using is actually just below 67M parameters. This should work ig.
# lol uses like 140 MB of VRAM. should have used some larger models but ig will try this before other big models depending on requirements.


def generate_response(query, reference_chunk, tokenizer, model, device):
    if scores[0] < 0.6:
        return "The reference provided is not relevant enough to generate a meaningful answer."
    
    # Prepare the input for BART in the correct format
    input_query = f"summarize the query '{query}' with context to: {reference_chunk}"
    
    # Tokenize input
    inputs = tokenizer(
        input_query,
        max_length=512,  # Limit to a max length to avoid memory issues
        truncation=True,  # Truncate input if it exceeds max_length
        return_tensors="pt",
        padding="max_length"
    ).to(device)
    
    # Use the model to generate an output (summary or answer)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=200,  # You can adjust the max_length for the output
            num_beams=5,  # Beam search for better generation quality
            early_stopping=True  # Stop early when a complete answer is found
        )
    
    # Decode the generated token IDs to a human-readable text
    generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Handle cases where the answer might be empty or just whitespace
    if not generated_text.strip():
        return "I couldn't generate a clear answer from the reference."
    
    return generated_text.strip()


a=0

while a!=2:
    print("\n\n1. Enter a query")
    print("2. Exit")
    try:
        a = int(input("Enter your choice: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    if a==1:
        query = input("Enter your query: ")
        # Get just the scores and indices of top related results
        scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
        # print_top_results_and_scores(query=query, embeddings=embeddings)

        if scores[1] > 0.6:
            references = f'{pages_and_chunks[indices[0]]["sentencechunk"]}\n{pages_and_chunks[indices[1]]["sentencechunk"]}'
        else:
            references = f'{pages_and_chunks[indices[0]]["sentencechunk"]}'
        # print(references)
        # print("---")
        print(generate_response(query=query, reference_chunk=references, tokenizer=tokenizer, model=llm_model, device=device))
    
    elif a==2:
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")
        continue


# big data analysis, Structured data and veracity as examples. 