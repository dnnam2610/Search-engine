import glob
import os
import json
from datasets import load_dataset
import sys
sys.path.append('../')


def read_wiki(dir):
    txt_files = glob.glob(os.path.join(dir, "*.txt"))  
    if not txt_files:
        raise ValueError(f"No .txt files found in {dir}")

    print(txt_files[0])
    dataset = load_dataset(
        path="text",
        data_files=txt_files,
        sample_by='document'
    )

    return dataset['train']


def clean_word(w):
  letters = set('aáàảãạăaáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz0123456789')
  new_w = ''
  for letter in w:
    if letter.lower() in letters or letter == '.':
      new_w += letter.lower()

  return new_w

def preprocess(docs):
    new_docs = []
    for doc in docs["text"]:  # docs là dict vì batched=True
        doc = doc.replace('\n', ' ').replace('==', ' ')
        words = doc.split()
        words = [clean_word(word) for word in words]
        new_doc = ' '.join(words)
        new_docs.append(new_doc)

    return {"text": new_docs}

def save_wiki(out_dir, docs):
   with open(out_dir, 'w') as f:
      json.dump(docs, f)
   


if __name__ == '__main__':
    dir = 'corpus.viwiki/viwiki'
    out_dir = 'processed_wiki.json'
    data = read_wiki(dir)
    processed_data = data.map(preprocess, batched=True)
    save_wiki(out_dir, processed_data.to_list())
    