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

        /**
         * Compute the Related Relaxed Word Mover's Distance (Rel-RWMD) between document 1 and document 2.
         * The idea is finding the closest token in D2 for each token in D1 as long as they are related.
         * Then do a weighted sum of the distances found and the respective weights of tokens in D1.
         *
         * @param nbow1 Map of tokens in Document 1 and their respective weights
         * @param nbow2 Map of tokens in Document 2 and their respective weights
         * @return Rel-RWMD distance
         */
        float computeDistance(const std::unordered_map<unsigned int, float>& nbow1,
                              const std::unordered_map<unsigned int, float>& nbow2) {
            // For a faster execution and tighter WMD approximation, documents do not share tokens.
            // E.g: If nbow1 and nbow2 contain the token 'AI' with nbow1['AI'] = 0.1 and nbow2['AI'] = 0.2.
            //      We take the diff so that 'AI' is removed from nbow1 and nbow2['AI'] = 0.1.
            //      This is the same of transporting weight = 0.1 with cost 0.
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);

            // Compute WMD relaxation - ignoring one constraint at a time
            float l12 = computeAsymmetricDistance(diffNbow1, diffNbow2);
            float l21 = computeAsymmetricDistance(diffNbow2, diffNbow1);
            return std::max(l12, l21);
        }

    private:

        std::shared_ptr<C_RelatedWords> m_relatedWords;

        /**
         * Compute the Asymmetrical Related Relaxed Word Mover's Distance (Rel-RWMD) from document 1 to document 2.
         * The idea is finding the closest token in D2 for each token in D1 as long as they are related.
         * Then do a weighted sum of the distances found and the respective weights of tokens in D1.
         *
         * @param nbow1 Map of tokens in Document 1 and their respective weights
         * @param nbow2 Map of tokens in Document 2 and their respective weights
         * @return Asymmetric Rel-RWMD distance
         */
        float computeAsymmetricDistance(const std::unordered_map<unsigned int, float>& nbow1,
                                        const std::unordered_map<unsigned int, float>& nbow2) {
            float value = 0.0f;
            for(const auto& it1: nbow1) {
                float weight = it1.second, minDistance = m_relatedWords->getMaximumDistance();

                // We want to find out the tokens in common between nbow2 and related words.
                // Since both related words and nbow2 are hashed, iterate over the smaller one of the two to get the
                // tokens in common. Using this approach, we should have have a faster execution.
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
