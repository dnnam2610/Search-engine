"""
Prepare data for searching
"""
from collections import defaultdict
import json
from datasets import load_dataset
from utils import get_idf, get_tfidf, get_tf, rounding
from tqdm import tqdm

class TFIDFDataGenerator:
    def __init__(self):
        """
        stats = {
            "words": {
                "quả": {"Doc1", "Doc2"}
                "bưởi": {"Doc1"}
            },
            "docs": {
                1: {
                    "quả": 1,
                    "bưởi": 1,
                    "ngon": 1
                },
                2: {
                    "quả": 1,
                    "táo": 1,
                    "dở": 1
                }
            }
        }

        tf_idf_list = {
            "quả":{
                0: 0.0
                1: 0.0
            }
            "táo":{
                0: 0.49
                1: 0.01
            }
            "quả":{
                0: 0.91
                1: 0.4
            }
        }

        self.ds = {
            0: 0.249,
            1: 0.249,
            2: 0.249
        }
        """
        self.stats = {
            'words':{},
            'docs':{}
        }
        self.tf_idf_list = defaultdict(lambda: defaultdict(float))
        self.ds = defaultdict(float)
    
    def generate(self, data_files):
        # Load data 
        self.dataset = load_dataset(
            path="json",
            data_files=data_files,
            split="train"
        )

        # Create stats index
        print("Generating stats index (stats)...")
        for i, example in enumerate(tqdm(self.dataset, desc="text processing")):
            text = example["text"]

            if i not in self.stats['docs']:
                self.stats['docs'][i] = defaultdict(int)

            for word in text.split(' '):
                word = word.strip()
                if not word:
                    continue

                if word not in self.stats['words']:
                    self.stats['words'][word] = {i}
                else:
                    self.stats['words'][word].add(i)

                self.stats['docs'][i][word] += 1

        # Compute TF-IDF
        words = self.stats['words'].keys()
        idf = defaultdict(float)
        N = len(self.dataset)

        # Compute IDF
        print("Computing IDF...")
        for word in tqdm(words, desc="IDF"):
            df = len(self.stats['words'][word])
            idf[word] = get_idf(N, df)

        # Compute TF-IDF for each document
        print("Computing TF-IDF...")
        for doc in tqdm(self.stats['docs'], desc="TF-IDF"):
            d = 0
            doc_length = sum(self.stats['docs'][doc].values())
            for word in self.stats['docs'][doc]:
                tf = get_tf(self.stats['docs'][doc][word], doc_length)

                tf_idf = tf * idf[word]
                d += tf_idf ** 2

                # Assign TF-IDF score to tf_idf_list[word][doc]
                self.tf_idf_list[word][doc] = tf_idf

            # Normalize vector length
            d_ = d ** 0.5

            # Save normalized length
            self.ds[doc] = rounding(d_)

    def save_tfidf_list(self, out_dir):
        with open(out_dir, 'w') as outfile:
            json.dump(self.tf_idf_list, outfile)

    def save_ds(self, out_dir):
        with open(out_dir, 'w') as outfile:
            json.dump(self.ds, outfile)

if __name__ == '__main__':
    generator = TFIDFDataGenerator()
    generator.generate('processed_wiki.json')
    generator.save_tfidf_list('tf_idf_list.json')
    generator.save_ds('ds.json')