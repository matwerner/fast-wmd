from nltk.corpus import stopwords
from urllib.parse import unquote

import html
import numpy
import string
import struct
import xmltodict

filepaths = [
	'hand/wikipedia-triplets-dump-20190918203353.xml'
]


# PREPROCESSING STEP
stop_words = set(stopwords.words('english'))
def preprocess_content(text):
	preprocessed_text = html.unescape(text)

	# Remove punctiation
	for sp in set(string.punctuation):
		preprocessed_text = preprocessed_text.replace(sp, ' ')

	# Remove numbers
	for number in '0123456789':
		preprocessed_text = preprocessed_text.replace(number, ' ')

	# Remove multiple whitespaces
	preprocessed_text = [w for w in preprocessed_text.split() if not w.lower() in stop_words]

	return preprocessed_text

# GET DOCUMENTS AND "HASH" TOKENS
token2id = dict()
id2token = []
url2id = dict()
texts = []
for filepath in filepaths:
	with open(filepath) as fp:
		print("FILE: ", filepath)

		doc = xmltodict.parse(fp.read())
		for page in doc['mediawiki']['page']:
			url = page['title'].replace(' ', '_') # .lower() # Some wikipedia pages have the same name - Differ only in case
			text = page['revision']['text']['#text']
			text = preprocess_content(text)

			hashed_text = []
			for token in text:
				if token not in token2id:
					token2id[token] = len(token2id)
					id2token.append(token)
				token_idx = token2id[token]
				hashed_text.append(token_idx)
			if url in url2id:
				print(filepath, url, url2id[url], len(url2id))
			url2id[url] = len(url2id)
			texts.append(hashed_text)

print("NUMBER OF URLS:", len(url2id))
print("NUMBER OF TEXTS:", len(texts))
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
with open("wikipedia-embeddings.txt", "w") as fp:
	fp.write("%d %d\n" % (len(embeddings), len(embeddings[0])))
	for tid, embedding in enumerate(embeddings):
		fp.write("%d %s\n" % (tid, ' '.join(embedding.astype(str))))

# DUMP VALID TOKENS
print("NUMBER OF TOKENS (EMBEDDINGS):", len(tokens))
with open("wikipedia-tokens.txt", mode='w', encoding='UTF-8') as fp:
	for idx, token in enumerate(tokens):
		fp.write("%s %d\n" % (token, idx))

# DUMP TRIPLETS IN INDEX FORMAT
with open("hand/wikipedia-triplets-release.txt") as fp, open("wikipedia-triplets.txt", mode='w') as fp_out:
	for line in fp:
		if not line.strip() or line.startswith('#'):
			continue
		indices = []
		for url_raw in line.strip().split(' '):
			url = url_raw.replace("http://en.wikipedia.org/wiki/", "")
			url = unquote(url)

			if url not in url2id:
				continue
			indices.append(str(url2id[url]))
		if len(indices) != 3:
			continue
		fp_out.write(' '.join(indices) + '\n')

# DUMP DOCUMENTS
print("NUMBER OF PAPERS:", len(texts))
with open("wikipedia-papers.txt", mode='w') as fp:
	for idx, text in enumerate(texts):
		new_text = [str(old_tid2new_tid[int(tid)]) 
					for tid in text if int(tid) in old_tid2new_tid]
		fp.write(' '.join(new_text) + '\n')
