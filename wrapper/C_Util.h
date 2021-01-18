#ifndef FAST_WMD_C_UTIL_H
#define FAST_WMD_C_UTIL_H

#include <vector>
#include <unordered_map>
#include "C_CONSTANTS.h"

namespace fastwmd {

    class C_Util {

    public:

        static std::vector<TokenIndex> getNbowIndices(const Document& nbow) {
            std::vector<TokenIndex> indices(nbow.size());
            for(std::size_t i = 0; i < nbow.size(); i++) {
                indices[i] = nbow[i].first;
            }
            return indices;
        }

        static Document diffNbow(const Document& nbow1, const Document& nbow2) {
            Document diffNbow;
            diffNbow.reserve(nbow1.size());

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
