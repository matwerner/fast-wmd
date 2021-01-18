# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

ctypedef unsigned int uint

# something.pxd
cdef extern from "limits.h":
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

cdef extern from "C_CONSTANTS.h" namespace "fastwmd":
    ctypedef unsigned int TokenIndex
    ctypedef float TokenWeight
    ctypedef float DistanceValue
    ctypedef vector[pair[TokenIndex,TokenWeight]] Document
    ctypedef unordered_map[TokenIndex, DistanceValue] HashedRelatedWords

# Declare the class with cdef
cdef extern from "C_Embeddings.h" namespace "fastwmd":
    cdef cppclass C_Embeddings:
        C_Embeddings() except +
        C_Embeddings(string) except +
        C_Embeddings(string, uint) except +
        uint getNumEmbeddings()
        uint getEmbeddingSize()
        vector[string] getTokens()

cdef extern from "C_RelatedWords.h" namespace "fastwmd":
    cdef cppclass C_RelatedWords:
        C_RelatedWords() except +
        C_RelatedWords(shared_ptr[C_Embeddings], uint) except +
        const HashedRelatedWords& getRelatedWords(TokenIndex)
        DistanceValue getMaximumDistance()

cdef extern from "C_Distance.h" namespace "fastwmd":
    cdef cppclass C_Distance:
        C_Distance() except +
        DistanceValue computeDistance(const Document&, const Document&)
        vector[vector[DistanceValue]] computeDistances(const vector[Document]&, const vector[Document]&)
        vector[vector[DistanceValue]] computeDistances(const vector[Document]&)

cdef extern from "C_WMD.h" namespace "fastwmd":
    cdef cppclass C_WMD(C_Distance):
        C_WMD() except +
        C_WMD(shared_ptr[C_Embeddings]) except +

cdef extern from "C_RWMD.h" namespace "fastwmd":
    cdef cppclass C_RWMD(C_Distance):
        C_RWMD() except +
        C_RWMD(shared_ptr[C_Embeddings]) except +

cdef extern from "C_RelWMD.h" namespace "fastwmd":
    cdef cppclass C_RelWMD(C_Distance):
        C_RelWMD() except +
        C_RelWMD(shared_ptr[C_RelatedWords]) except +

cdef extern from "C_RelRWMD.h" namespace "fastwmd":
    cdef cppclass C_RelRWMD(C_Distance):
        C_RelRWMD() except +
        C_RelRWMD(shared_ptr[C_RelatedWords]) except +