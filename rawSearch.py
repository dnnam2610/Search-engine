import json
from datasets import load_dataset

class TFIDF():
    def __init__(self, tfidf_dir, ds_dir):
        # Load data
        with open(tfidf_dir, 'r') as f:
            self.tf_idf_list = json.load(f)
        with open(ds_dir, 'r') as f:
            self.ds = json.load(f)
    
    def search(self, q, top_k):
        # Search documents using TF-IDF
        results = []
        for i in range(len(self.ds.keys())):
            i = str(i)
            if self.ds[i] == 0:
                continue

            score = 0
            for t in q.split():
                t = t.lower()
                # tf_idf_list["miền"][i]
                # tf_idf_list["trung"][i]
                score += self.tf_idf_list.get(t, {}).get(i, 0) / self.ds[i]
            results.append((score, i))
        results = sorted(results, key= lambda x: x[0], reverse=True)[:top_k]
        return results
    

# if __name__ == "__main__":
    # tfidf = TFIDF(
    #     tfidf_dir='storage/tf_idf_list.json',
    #     ds_dir='storage/ds.json'
    # )

    # query = 'miền Bắc'

    # docs = load_dataset(
    #     path='json',
    #     data_files='storage/processed_wiki.json',
    #     split='train'
    # )

    # results = tfidf.search(query, 5)
    # print(results)
    # print(docs[results[0][1]])