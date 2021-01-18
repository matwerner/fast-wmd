#ifndef FAST_WMD_C_UTIL_H
#define FAST_WMD_C_UTIL_H

#include <vector>
#include <unordered_map>
#include "C_CONSTANTS.h"

namespace fastwmd {

    class C_Util {

    public:

        /**
         * Get the token indices in a given document
         * 
         * @param nbow L1-Normalized BOW representation of document (sorted by token index)
         * @return List of token indices in document
         */
        static std::vector<TokenIndex> getNbowIndices(const Document& nbow) {
            std::vector<TokenIndex> indices(nbow.size());
            for(std::size_t i = 0; i < nbow.size(); i++) {
                indices[i] = nbow[i].first;
            }
            return indices;
        }

        /**
         * Get a BOW representation from the Doc1 - Doc2 (Only non-negative weights)
         * 
         * @param nbow1 L1-Normalized BOW representation of Doc1 (sorted by token index)
         * @param nbow2 L1-Normalized BOW representation of Doc2 (sorted by token index)
         * @return BOW representation from the Doc1 - Doc2 (Only non-negative weights)
         */
        static Document diffNbow(const Document& nbow1, const Document& nbow2) {
            Document diffNbow;
            diffNbow.reserve(nbow1.size());

            // Linear time comparison between lists -> Doc1 and Doc2 must be sorted
            std::size_t i = 0, j = 0;
            while(i < nbow1.size()) {
                if(j >= nbow2.size() || nbow1[i].first < nbow2[j].first) {
                    diffNbow.emplace_back(nbow1[i]);
                    i++;
                } else if(nbow1[i].first == nbow2[j].first) {
                    TokenWeight weight = nbow1[i].second - nbow2[j].second;
                    if(weight > 0) diffNbow.emplace_back(nbow1[i].first, weight);
                    i++; j++;
                } else {
                    j++;
                }
            }
            return diffNbow;
        }

        /**
         * Convert BOW representation data structure used from vector to hashed map
         * 
         * @param nbow L1-Normalized BOW representation of document
         * @return L1-Normalized BOW representation of Doc2 (using hashed map)
         */
        static HashedDocument getHashedDocument(const Document& nbow) {
            HashedDocument hashedNbow;
            hashedNbow.reserve(nbow.size());
            for(std::size_t i = 0; i < nbow.size(); i++) {
                hashedNbow[nbow[i].first] = nbow[i].second;
            }
            return hashedNbow;
        }

    };
}


#endif //FAST_WMD_C_UTIL_H
