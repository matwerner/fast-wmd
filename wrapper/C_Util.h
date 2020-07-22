#ifndef FAST_WMD_C_UTIL_H
#define FAST_WMD_C_UTIL_H

#include <vector>
#include <unordered_map>

namespace fastwmd {

    class C_Util {

    public:

        static std::vector<unsigned int> getNbowIndices(const std::vector<std::pair<unsigned int, float>>& nbow) {
            std::vector<unsigned int> indices(nbow.size());
            for(unsigned int i = 0; i < nbow.size(); i++) {
                indices[i] = nbow[i].first;
            }
            return indices;
        }

        static std::vector<std::pair<unsigned int, float>> diffNbow(const std::vector<std::pair<unsigned int, float>>& nbow1,
                                                                    const std::vector<std::pair<unsigned int, float>>& nbow2) {
            std::vector<std::pair<unsigned int, float>> diffNbow;
            diffNbow.reserve(nbow1.size());

            unsigned int i = 0, j = 0;
            while(i < nbow1.size()) {
                if(j >= nbow2.size() || nbow1[i].first < nbow2[j].first) {
                    diffNbow.emplace_back(nbow1[i]);
                    i++;
                } else if(nbow1[i].first == nbow2[j].first) {
                    float weight = nbow1[i].second - nbow2[j].second;
                    if(weight > 0) diffNbow.emplace_back(nbow1[i].first, weight);
                    i++; j++;
                } else {
                    j++;
                }
            }
            return diffNbow;
        }

        static std::unordered_map<unsigned int, float> diffNbow(const std::unordered_map<unsigned int, float>& nbow1,
                                                                const std::unordered_map<unsigned int, float>& nbow2) {
            std::unordered_map<unsigned int, float> diffNbow;
            diffNbow.reserve(nbow1.size());
            for(const auto& it: nbow1) {
                float weight = it.second;
                // Check whether token is in both documents
                const auto& itCommon = nbow2.find(it.first);
                weight -= itCommon == nbow2.end()? 0 :  itCommon->second;
                if(weight <= 0) continue;
                diffNbow[it.first] = weight;
            }
            return diffNbow;
        }

    };
}


#endif //FAST_WMD_C_UTIL_H
