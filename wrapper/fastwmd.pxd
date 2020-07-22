# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

ctypedef unsigned int uint
ctypedef vector[pair[uint,float]] standard_nbow
ctypedef vector[vector[pair[uint,float]]] standard_nbow_list
ctypedef unordered_map[uint,float] hashed_nbow
ctypedef vector[unordered_map[uint,float]] hashed_nbow_list

# something.pxd
cdef extern from "limits.h":
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

# Declare the class with cdef
cdef extern from "C_Embeddings.h" namespace "fastwmd":
    cdef cppclass C_Embeddings:
        C_Embeddings() except +
        C_Embeddings(string) except +
        C_Embeddings(string, uint) except +
        int getNumEmbeddings()
        int getEmbeddingSize()
        vector[string] getTokens()

cdef extern from "C_RelatedWords.h" namespace "fastwmd":
    cdef cppclass C_RelatedWords:
        C_RelatedWords() except +
        C_RelatedWords(shared_ptr[C_Embeddings], uint) except +
        const unordered_map[uint, float]& getRelatedWords(uint)
        float getMaximumDistance()

cdef extern from "C_Distance.h" namespace "fastwmd":
    cdef cppclass C_Distance[T]:
        C_Distance() except +
        float computeDistance(const T&, const T&)
        vector[vector[float]] computeDistances(const vector[T]&, const vector[T]&)
        vector[vector[float]] computeDistances(const vector[T]&)

cdef extern from "C_WMD.h" namespace "fastwmd":
    cdef cppclass C_WMD(C_Distance[standard_nbow]):
        C_WMD() except +
        C_WMD(shared_ptr[C_Embeddings]) except +

cdef extern from "C_RWMD.h" namespace "fastwmd":
    cdef cppclass C_RWMD(C_Distance[standard_nbow]):
        C_RWMD() except +
        C_RWMD(shared_ptr[C_Embeddings]) except +

cdef extern from "C_RelWMD.h" namespace "fastwmd":
    cdef cppclass C_RelWMD(C_Distance[hashed_nbow]):
        C_RelWMD() except +
        C_RelWMD(shared_ptr[C_RelatedWords]) except +

cdef extern from "C_RelRWMD.h" namespace "fastwmd":
    cdef cppclass C_RelRWMD(C_Distance[hashed_nbow]):
        C_RelRWMD() except +
        C_RelRWMD(shared_ptr[C_RelatedWords]) except +