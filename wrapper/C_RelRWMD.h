#ifndef FAST_WMD_C_RELRWMD_H
#define FAST_WMD_C_RELRWMD_H


#include <memory>
#include "C_Distance.h"
#include "C_RelatedWords.h"
#include "C_Util.h"

namespace fastwmd {

    class C_RelRWMD: public C_Distance<std::unordered_map<unsigned int, float>> {

    public:

        C_RelRWMD() {}

        C_RelRWMD(const std::shared_ptr<C_RelatedWords>& relatedWords): m_relatedWords(relatedWords) {}

        float computeDistance(const std::unordered_map<unsigned int, float>& nbow1,
                              const std::unordered_map<unsigned int, float>& nbow2) {
            // Documents do not share tokens
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);
            float l12 = computeAsymmetricDistance(diffNbow1, diffNbow2);
            float l21 = computeAsymmetricDistance(diffNbow2, diffNbow1);
            return std::max(l12, l21);
        }

    private:

        std::shared_ptr<C_RelatedWords> m_relatedWords;

        float computeAsymmetricDistance(const std::unordered_map<unsigned int, float>& nbow1,
                                        const std::unordered_map<unsigned int, float>& nbow2) {
            float value = 0.0f;
            for(const auto& it1: nbow1) {
                float weight = it1.second, minDistance = m_relatedWords->getMaximumDistance();

                // Whether is better to compare doc2 -> related words or the opposite
                const auto& relatedWords = m_relatedWords->getRelatedWords(it1.first);
                if(nbow2.size() < relatedWords.size()) {
                    for (const auto& it2: nbow2) {
                        const auto& itR = relatedWords.find(it2.first);
                        if(itR == relatedWords.end()) continue;
                        minDistance = std::min(minDistance, itR->second);
                    }
                } else {
                    for(const auto& itR: relatedWords) {
                        const auto& it2 = nbow2.find(itR.first);
                        if(it2 == nbow2.end()) continue;
                        minDistance = std::min(minDistance, itR.second);
                    }
                }
                value += weight * minDistance;
            }
            return value;
        }
    };

}


#endif //FAST_WMD_C_RELRWMD_H
