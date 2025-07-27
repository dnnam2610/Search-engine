"""
This file is used to combine multi wiki files into one
"""

from utils import read_wiki, save_wiki, preprocess

if __name__ == '__main__':
    dir = 'corpus.viwiki/viwiki'
    out_dir = 'processed_wiki.json'
    data = read_wiki(dir)
    processed_data = data.map(preprocess, batched=True)
    save_wiki(out_dir, processed_data.to_list())