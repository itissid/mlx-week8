from functools import partial
import hashlib
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# import pandas as pd
# from tqdm import tqdm
# import torch


def create_lookups(dataset):
    print(f"dataset shape: {dataset.shape}")
    all_urls = dataset["passages"].apply(lambda x: x["url"]).tolist()

    unique_urls = set([item for sublist in all_urls for item in sublist])
    print(f"Total number of urls: {sum(len(i) for i in all_urls)}")
    print(f"Total number of unique urls: {len(unique_urls)}")

    # Use an md5 hash for the urls for deterministic mapping
    def generate_md5_hash(s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    ids_to_urls = {generate_md5_hash(url): url for url in unique_urls}

    urls_to_ids = {url: i for i, url in ids_to_urls.items()}
    print(f"Total number of hashed urls: {len(ids_to_urls)}")

    url_id_to_doc_mapping = {}

    for row in dataset[["passages"]].iterrows():
        # assert len(url_list) == len(set(url_list))
        passages = row[1]["passages"]
        for url, passage_text in zip(passages["url"], passages["passage_text"]):
            url_id_to_doc_mapping[urls_to_ids[url]] = passage_text

    query_ids = dataset["query_id"].tolist()
    assert len(query_ids) == len(set(query_ids))
    print(f"Total number of queries: {len(query_ids)}")
    query_id_to_query_mapping = {}
    query_id_to_wfa = {}
    for _, row in dataset[["query_id", "query", "wellFormedAnswers"]].iterrows():
        query_id = row["query_id"]
        query = row["query"]
        well_formed_answers = row["wellFormedAnswers"][0]
        query_id_to_query_mapping[query_id] = query
        query_id_to_wfa[query_id] = well_formed_answers

    return (
        ids_to_urls,
        urls_to_ids,
        url_id_to_doc_mapping,
        query_id_to_query_mapping,
        query_id_to_wfa,
    )


def add_hashed_urls(dataset, urls_to_ids):
    dataset.loc[:, "hashed_urls"] = dataset["passages"].progress_apply(
        lambda x: np.array(list(set([urls_to_ids[url] for url in x["url"]])))
    )


def populate_embeddings_for_passages(
    bert_encoder, bert_tokenizer, url_to_doc_mapping, batch_size=32
):

    pass


from transformers import RobertaModel
import torch
from tqdm import tqdm

def generate_embeddings(encoded_inputs, batch_size=200, device="cpu"):
    all_embeddings = []
    
    model = RobertaModel.from_pretrained("roberta-base")
    model.to(device)
    # Process input data in batches
    num_passages = encoded_inputs["input_ids"].shape[0]
    encoded_inputs['input_ids'] = encoded_inputs["input_ids"].to(device)
    encoded_inputs['attention_mask']  =  encoded_inputs["attention_mask"].to(device)
    # We don't need a dataloader here because all the data is in GPU already.
    for i in tqdm(range(0, num_passages, batch_size)):
        inputs = dict(
            input_ids=encoded_inputs["input_ids"][i : i + batch_size, :],
            attention_mask=encoded_inputs["attention_mask"][i : i + batch_size, :],
        )

        # Tokenize the batch
        # inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate embeddings for the batch
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings for the `[CLS]` token
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().clone()
       
        # Append the embeddings to the list
        all_embeddings.append(cls_embeddings)
        del outputs
        del inputs

    # with torch.no_grad():
    # outputs = model(**encoded_inputs)
    return torch.cat(all_embeddings, dim=0) # Extracting the [CLS] token's embeddings
