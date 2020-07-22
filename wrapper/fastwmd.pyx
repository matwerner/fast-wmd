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

    def get_related_words(self, uint index):
        return self.c_related_words.get().getRelatedWords(index)

    def get_maximum_distance(self):
        return self.c_related_words.get().getMaximumDistance()

# Reference:
# https://stackoverflow.com/questions/28573479/cython-python-c-inheritance-passing-derived-class-as-argument-to-function-e
cdef class StandardDistance:
    cdef C_Distance[standard_nbow]* c_distance

    def compute_distance(self, const standard_nbow& nbow1, const standard_nbow& nbow2):
        return self.c_distance.computeDistance(nbow1, nbow2)

    def compute_distances(self, const standard_nbow_list& nbows1, const standard_nbow_list& nbows2=[]):
        if len(nbows2) == 0:
            return self.c_distance.computeDistances(nbows1)
        else:
            return self.c_distance.computeDistances(nbows1, nbows2)

cdef class WMD(StandardDistance):

    def __cinit__(self, Embeddings embeddings):
        self.c_distance = new C_WMD(embeddings.c_embeddings)

cdef class RWMD(StandardDistance):

    def __cinit__(self, Embeddings embeddings):
        self.c_distance = new C_RWMD(embeddings.c_embeddings)

cdef class HashedDistance:
    cdef C_Distance[hashed_nbow]* c_distance

    def compute_distance(self, const hashed_nbow& nbow1, const hashed_nbow& nbow2):
        return self.c_distance.computeDistance(nbow1, nbow2)

    def compute_distances(self, const hashed_nbow_list& nbows1, const hashed_nbow_list& nbows2=[]):
        if len(nbows2) == 0:
            return self.c_distance.computeDistances(nbows1)
        else:
            return self.c_distance.computeDistances(nbows1, nbows2)

cdef class RelWMD(HashedDistance):

    def __cinit__(self, RelatedWords related_words):
        self.c_distance = new C_RelWMD(related_words.c_related_words)

cdef class RelRWMD(HashedDistance):

    def __cinit__(self, RelatedWords related_words):
        self.c_distance = new C_RelRWMD(related_words.c_related_words)
