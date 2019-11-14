import json
import numpy
import os.path
import scipy.io
import sys
import urllib.request

# Dataset availables
dataset_paths = {"amazon": "amazon-emd_tr_te_split.mat",
				 "bbcsport": "bbcsport-emd_tr_te_split.mat",
				 "classic": "classic-emd_tr_te_split.mat",
				 "recipe": "recipe2-emd_tr_te_split.mat",
				 "twitter": "twitter-emd_tr_te_split.mat"}

# Links at https://github.com/mkusner/wmd
dataset_links = {"amazon": "https://www.dropbox.com/sh/nf532hddgdt68ix/AABNO4w-1T6ozVLCxrRNjCgGa/amazon-emd_tr_te_split.mat?dl=1",
				 "bbcsport": "https://www.dropbox.com/sh/nf532hddgdt68ix/AAArRFEToUmSJg8G9v120rBQa/bbcsport-emd_tr_te_split.mat?dl=1",
				 "classic": "https://www.dropbox.com/sh/nf532hddgdt68ix/AABLYdubbTlj5-VgiCNKTTopa/classic-emd_tr_te_split.mat?dl=1",
				 "recipe": "https://www.dropbox.com/sh/nf532hddgdt68ix/AAAbRqPilTT2Yn9EO-eqsPdOa/recipe2-emd_tr_te_split.mat?dl=1",
				 "twitter": "https://www.dropbox.com/sh/nf532hddgdt68ix/AABg21VwvpVlEXkAGwf4c3bxa/twitter-emd_tr_te_split.mat?dl=1"}

# Get console args
dataset_name = sys.argv[1]
dataset_index = int(sys.argv[2])

if dataset_name not in dataset_paths:
	print("Dataset availables:")
	for dataset_name in dataset_paths.keys():
		print(dataset_name)
	sys.exit(0)

if dataset_index < 0 or dataset_index > 4:
	print("Dataset indices go from 0 to 4")
	sys.exit(0)

# Parse console args
dataset_path = os.path.join(dataset_name, dataset_paths[dataset_name])
if not os.path.exists(dataset_path):
	dataset_link = dataset_links[dataset_name]
	os.mkdir(dataset_name)
	urllib.request.urlretrieve(dataset_link, dataset_path)	
dataset = scipy.io.loadmat(dataset_path)

# Get tokens of each document
texts = []
labels = []
embeddings = []
token2id = dict()
id2token = []

if dataset_name != "recipe":	
	for doc_embeddings, label, bow, doc in zip(dataset['X'][0], dataset['Y'][0], dataset['BOW_X'][0], dataset['words'][0]):
		text = []
		for doc_embedding, freq, word in zip(doc_embeddings.T, bow[0], doc[0]):
			token = word[0]
			if token not in token2id:
				embeddings.append(doc_embedding)
				token2id[token] = len(token2id)
				id2token.append(token)
			tokenId = token2id[token]
			text += freq * [str(tokenId)]
		texts += [text]
		labels += [int(label)-1]
else:
	"""
	Recipe has a some problems regarding dataset['words']
	1. 	The key is renamed to 'the_words'
	2. 	For unknown motives, each time a null entry array appears, the following values in dataset['words'] are shift one position.
		E.g: If we got dataset['words'][1399] == [], the value of 1399 will be storaged in dataset['words'][1399 + 1], the same for all values after it.
			Then if we got dataset['words'][2000 + 1] == [], the value of 2000 will be storaged in dataset['words'][2000 + 2]

		Unfortunately, this means the last entries do not have any value (as the dataset has ended), so we have to try retrieving it from the embeddings
	"""
	size = len(dataset['X'][0])
	p = 0 # Padding
	for i in range(size):
		if len(dataset['the_words'][0][i+p]) == 0:
			print(p)
			p+=1

		doc_embeddings, label, bow = dataset['X'][0][i].T, dataset['Y'][0][i], dataset['BOW_X'][0][i][0]

		if i + p < size:
			doc = dataset['the_words'][0][i+p][0]
		else:
			doc, to_be_removed = [], []
			for j, doc_embedding in enumerate(doc_embeddings):			
				found = False
				for t, e in enumerate(embeddings):
					if numpy.allclose(doc_embedding, e):
						found = True
						doc += [id2token[t]]
						break
				if not found:
					to_be_removed += [j]

			doc = numpy.delete(numpy.array(doc), to_be_removed)
			doc_embeddings = numpy.delete(doc_embeddings, to_be_removed, 0) 
			bow = numpy.delete(bow, to_be_removed)		

		text = []
		for doc_embedding, freq, word in zip(doc_embeddings, bow, doc):
			token = word[0]			
			if token not in token2id:
				embeddings.append(doc_embedding)
				token2id[token] = len(token2id)
				id2token.append(token)
			tokenId = token2id[token]
			text += freq * [str(tokenId)]
		texts += [text]
		labels += [int(label)-1]

# Split dataset in training and test
documents_tr = [(labels[idx-1], texts[idx-1]) for idx in dataset['TR'][dataset_index]]
documents_te = [(labels[idx-1], texts[idx-1]) for idx in dataset['TE'][dataset_index]]

# Dump dataset path
documents_path_format = os.path.join(dataset_name, "{0}-{1}-{2}.txt")
documents_tr_path = documents_path_format.format(dataset_name, 'train', dataset_index)
documents_te_path = documents_path_format.format(dataset_name, 'test', dataset_index)

# Dump dataset
line_format = "{0} {1}\n"
with open(documents_tr_path, mode='w', encoding='UTF-8') as fp:	
	for document in documents_tr:
		label, text = document
		fp.write(line_format.format(label, ' '.join(text)))
with open(documents_te_path, mode='w', encoding='UTF-8') as fp:
	for document in documents_te:
		label, text = document
		fp.write(line_format.format(label, ' '.join(text)))

# Dump word embeddings - Gensim format
embeddings_path = os.path.join(dataset_name, dataset_name + "-embeddings.txt")
with open(embeddings_path, mode='w', encoding='UTF-8') as fp:		
	fp.write("%d %d\n" % (len(embeddings), len(embeddings[0])))
	for token, embedding in enumerate(embeddings):
		fp.write("%s %s\n" % (token, ' '.join(embedding.astype(str))))

# Dump tokens index
tokens_path = os.path.join(dataset_name, dataset_name + "-tokens.txt")
with open(tokens_path, mode='w', encoding='UTF-8') as fp:		
	for idx, token in enumerate(id2token):
		fp.write("%s %d\n" % (token, idx))