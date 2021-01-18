# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
cimport fastwmd

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class Embeddings:
    cdef shared_ptr[C_Embeddings] c_embeddings  # Hold a C++ instance which we're wrapping

    def __cinit__(self, string filepath, uint max_num_embeddings=UINT_MAX):
        if max_num_embeddings == UINT_MAX:
            self.c_embeddings.reset(new C_Embeddings(filepath))
        else:
            self.c_embeddings.reset(new C_Embeddings(filepath, max_num_embeddings))

    def get_num_embeddings(self):
        return self.c_embeddings.get().getNumEmbeddings()

    def get_embedding_size(self):
        return self.c_embeddings.get().getEmbeddingSize()

    def get_tokens(self):
        return self.c_embeddings.get().getTokens()

cdef class RelatedWords:
    cdef shared_ptr[C_RelatedWords] c_related_words

    def __cinit__(self, Embeddings embeddings, uint r):
        self.c_related_words.reset(new C_RelatedWords(embeddings.c_embeddings, r))

    def get_related_words(self, TokenIndex index):
        return self.c_related_words.get().getRelatedWords(index)

    def get_maximum_distance(self):
        return self.c_related_words.get().getMaximumDistance()

# Reference:
# https://stackoverflow.com/questions/28573479/cython-python-c-inheritance-passing-derived-class-as-argument-to-function-e
cdef class Distance:
    cdef C_Distance* c_distance

    def compute_distance(self, const Document& nbow1, const Document& nbow2):
        return self.c_distance.computeDistance(nbow1, nbow2)

    def compute_distances(self, const vector[Document]& nbows1, const vector[Document]& nbows2=[]):
        if len(nbows2) == 0:
            return self.c_distance.computeDistances(nbows1)
        else:
            return self.c_distance.computeDistances(nbows1, nbows2)

cdef class WMD(Distance):

    def __cinit__(self, Embeddings embeddings):
        self.c_distance = new C_WMD(embeddings.c_embeddings)

cdef class RWMD(Distance):

    def __cinit__(self, Embeddings embeddings):
        self.c_distance = new C_RWMD(embeddings.c_embeddings)

cdef class RelWMD(Distance):

    def __cinit__(self, RelatedWords related_words):
        self.c_distance = new C_RelWMD(related_words.c_related_words)

cdef class RelRWMD(Distance):

    def __cinit__(self, RelatedWords related_words):
        self.c_distance = new C_RelRWMD(related_words.c_related_words)
