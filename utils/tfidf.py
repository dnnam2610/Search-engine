import math

def rounding(num):
    return math.floor(num * 1000) / 1000
        
def get_tf(num, doc_length):
    return rounding(num/doc_length)

def get_idf(N, df):
    return math.log10(N/df)

def get_tfidf(tf, idf):
    return tf*idf