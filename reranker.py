
# import torch
# torch.set_num_threads(1)
from datasets import Dataset, load_from_disk, load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss
from rawSearch import TFIDF
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# model = AutoModel.from_pretrained('vinai/phobert-base')
# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

EMBEDDING_DOCS_PATH = 'storage/wiki_with_vec_dataset'
model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

# model = SentenceTransformer('Cloyne/vietnamese-sbert-v3')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, trust_remote_code=True,
    torch_dtype=torch.float16
)

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[MEMORY] {note} Memory usage: {mem:.2f} MB")

def get_embedding(item):
    embeddings = model.encode(item)
    # lấy vector embedidng của mẫu
    return embeddings


class ReRanker():
    def __init__(self):
        self.docs_dataset = load_from_disk(dataset_path=EMBEDDING_DOCS_PATH)

    def rank(self, query, raw_results):
        pairs = []
        for score, index in raw_results:
            pairs.append([self.docs_dataset['text'][index], query])
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            max_index = torch.argmax(scores).item()
            # print("Max score index:", max_index)
            # print("Best: ", pairs[max_index][0])
        
        return pairs[max_index][0]
    
if __name__ == '__main__':
    top_k = 1
    rawSearcher = TFIDF(
        tfidf_dir='storage/tf_idf_list.json',
        ds_dir='storage/ds.json'
    )

    reranker = ReRanker()
    query = 'yên bái'

    raw_results = rawSearcher.search(query, 5)
    result = reranker.rank(query, raw_results)
    print(result)


