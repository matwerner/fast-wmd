import json
import numpy
import os.path
import scipy.io
import sys
import urllib.request

# Dataset availables
dataset_paths = {"ohsumed": "ohsumed-emd_tr_te_ix.mat",
				 "reuters": "r8-emd_tr_te3.mat",
				 "20news":  "20ng2_500-emd_tr_te.mat"}

# Links at https://github.com/mkusner/wmd
dataset_links = {"ohsumed": "https://www.dropbox.com/sh/nf532hddgdt68ix/AAD31LaD0o04z7YSdxsfmZIca/ohsumed-emd_tr_te_ix.mat?dl=1",
				 "reuters": "https://www.dropbox.com/sh/nf532hddgdt68ix/AAD6gtc0gtB4n7zWwwnuPqrYa/r8-emd_tr_te3.mat?dl=1",
				 "20news":  "https://www.dropbox.com/sh/nf532hddgdt68ix/AACtd7NIdxXUfrxSvP-OUci4a/20ng2_500-emd_tr_te.mat?dl=1"}

# Get console args
dataset_name = sys.argv[1]

if dataset_name not in dataset_paths:
	print("Dataset availables:")
	for dataset_name in dataset_paths.keys():
		print(dataset_name)
	sys.exit(0)

# Parse console args
dataset_path = os.path.join(dataset_name, dataset_paths[dataset_name])
if not os.path.exists(dataset_path):
	dataset_link = dataset_links[dataset_name]
	os.mkdir(dataset_name)
	urllib.request.urlretrieve(dataset_link, dataset_path)	
dataset = scipy.io.loadmat(dataset_path)

embeddings = []
token2id = dict()
id2token = []

def getDatasetSplit(X, Y, BOW_X, words):
	# Get tokens of each document
	texts = []
	labels = []	
	error_count = 0
	for doc_embeddings, label, bow, doc in zip(X[0], Y[0], BOW_X[0], words[0]):
		text = []
		if len(doc_embeddings) == 0 or len(bow[0]) == 0 or len(doc[0]) == 0:
			# Should only occur once in 20news and none in the others
			print(error_count)
			error_count+=1
			continue
		for doc_embedding, freq, word in zip(doc_embeddings.T, bow[0].astype(numpy.int64), doc[0]):
			token = str(word[0])
			if token not in token2id:
				embeddings.append(doc_embedding)
				token2id[token] = len(token2id)
				id2token.append(token)
			tokenId = token2id[token]
			text += freq * [str(tokenId)]
		texts += [text]
		labels += [int(label)-1]

	# Create dataset
	documents = [(label, text) for label, text in zip(labels, texts)]
	return documents

# 20news does not have words - create words from embeddings
if dataset_name == "20news":
	count = 0
	hash_table = dict()
	for partition in ['tr', 'te']:
		dataset['words_' + partition] = [[]]
		for doc_embeddings in dataset['x' + partition][0]:
			doc = []
			for doc_embedding in doc_embeddings.T:
				key = hash(doc_embedding.tobytes())
				if key not in hash_table:
					hash_table[key] = []
				found = False
				for word, doc_embedding2 in hash_table[key]:
					if numpy.allclose(doc_embedding, doc_embedding2):
						doc.append([word])
						found = True
						break
				if not found:
					word = str(count)
					doc.append([word])
					hash_table[key].append((count, doc_embedding))
					count+=1
			dataset['words_' + partition][0].append([doc])

# Split dataset in training and test
documents_tr = getDatasetSplit(dataset['xtr'], dataset['ytr'], dataset['BOW_xtr'], dataset['words_tr'])
documents_te = getDatasetSplit(dataset['xte'], dataset['yte'], dataset['BOW_xte'], dataset['words_te'])

# Dump dataset path
documents_path_format = os.path.join(dataset_name, "{0}-{1}.txt")
documents_tr_path = documents_path_format.format(dataset_name, 'train')
documents_te_path = documents_path_format.format(dataset_name, 'test')

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

tokens_path = os.path.join(dataset_name, dataset_name + "-tokens.txt")
with open(tokens_path, mode='w', encoding='UTF-8') as fp:		
	for idx, token in enumerate(id2token):
		fp.write("%s %d\n" % (token, idx))