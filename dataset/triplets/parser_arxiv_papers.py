from nltk.corpus import stopwords

import ast
import json
import numpy
import os
import random
import string
import struct
import unidecode

# PREPROCESSING STEP
stop_words = set(stopwords.words('english'))
def preprocess_content(text):
    preprocessed_text = text
    
    # Remove punctiation
    for sp in set(string.punctuation):
        preprocessed_text = preprocessed_text.replace(sp, ' ')
        
    # Remove numbers
    for number in '0123456789':
        preprocessed_text = preprocessed_text.replace(number, ' ')
    
    # Remove multiple whitespaces
    preprocessed_text = [w for w in preprocessed_text.split() if not w in stop_words]
    
    return preprocessed_text

# GET DOCUMENTS AND "HASH" TOKENS
token2id = dict()
id2token = []
url2id = dict()
texts = []
for i in range(5):
    print("PART", i)

    filepath_template = "arxiv_papers_raw_{0}.tsv"
    for line in open(filepath_template.format(i)):
        if not line.strip():
            continue

        url, response = line.split('\t', 1)
        response = ast.literal_eval(response)
        if response["status"] == 200:
            text = preprocess_content(response["content"])
        else:
            print(line)
            continue

        hashed_text = []
        for token in text:
            if token not in token2id:
                token2id[token] = len(token2id)
                id2token.append(token)
            token_idx = token2id[token]
            hashed_text.append(token_idx)
        url2id[url] = len(url2id)
        texts.append(hashed_text)

print("NUMBER OF TOKENS (RAW):", len(id2token))

# GET VALID (IN DOCUMENTS) TOKENS FROM WORD2VEC DATASET
tokens = []
embeddings_batch = []
embeddings = []
old_tid2new_tid = dict()
word_embeddings_path = "<PATH-TO-EMBEDDINGS>/GoogleNews-vectors-negative300.bin"

# REFERENCE TO READ WORD2VEC BINARY FILE: https://gist.github.com/ottokart/4031dfb471ad5c11d97ad72cbc01b934
FLOAT_SIZE = 4
with open(word_embeddings_path, 'rb') as fp: #, open("arxiv-embeddings.txt", "w") as fp_out:
    # Read the header
    header = fp.readline().strip()

    total_num_vectors, vector_len = (int(x) for x in header.split())
    for j in range(total_num_vectors):
        
        word = b""        
        while True:
            c = fp.read(1)
            if c == b" " or c is None:
                break
            word += c
        # End of file
        if c is None:
            break        

        binary_vector = fp.read(FLOAT_SIZE * vector_len)

        token = word.decode('ISO-8859-1')        
        if token not in token2id:
            continue

        vector = [ struct.unpack_from('f', binary_vector, i)[0] 
                   for i in range(0, len(binary_vector), FLOAT_SIZE) ]        
                
        tid = len(tokens)
        old_tid2new_tid[token2id[token]] = tid
        embedding = numpy.array(vector)
        embedding = embedding / numpy.linalg.norm(embedding)
        tokens.append(token)
        embeddings.append(embedding)

# DUMP EMBEDDINGS FROM VALID TOKENS
print("NUMBER OF TOKENS (EMBEDDINGS):", len(tokens))
with open("arxiv-embeddings.txt", "w") as fp:
    fp.write("%d %d\n" % (len(embeddings), len(embeddings[0])))
    for tid, embedding in enumerate(embeddings):
        fp.write("%d %s\n" % (tid, ' '.join(embedding.astype(str))))

# DUMP VALID TOKENS
print("NUMBER OF TOKENS (EMBEDDINGS):", len(tokens))
with open("arxiv-tokens.txt", mode='w', encoding='UTF-8') as fp:
    for idx, token in enumerate(tokens):
        fp.write("%s %d\n" % (token, idx))

# DUMP TRIPLETS IN INDEX FORMAT
with open("arxiv_2014_09_27_examples.txt") as fp, open("arxiv-triplets.txt", mode='w') as fp_out:
    for line in fp:
        url1, url2, url3 = line.strip().split(' ')
        if url1 not in url2id or url2 not in url2id or url3 not in url2id:
            continue
        indices = [str(url2id[url]) for url in [url1, url2, url3]]
        fp_out.write(' '.join(indices) + '\n')

# DUMP DOCUMENTS
print("NUMBER OF PAPERS:", len(texts))
with open("arxiv-papers.txt", mode='w') as fp:
    for idx, text in enumerate(texts):
        new_text = [str(old_tid2new_tid[int(tid)]) 
                    for tid in text if int(tid) in old_tid2new_tid]
        fp.write(' '.join(new_text) + '\n')