#ifndef FAST_WMD_C_RWMD_H
#define FAST_WMD_C_RWMD_H

#include <memory>
#include "C_Distance.h"
#include "C_Embeddings.h"
#include "C_Util.h"

namespace fastwmd {

    class C_RWMD: public C_Distance {

    public:

        C_RWMD() {}

        C_RWMD(const std::shared_ptr<C_Embeddings>& embeddings): m_embeddings(embeddings) {}

        /**
         * Compute the Relaxed Word Mover's Distance (RWMD) from Doc1 to Doc2.
         * The idea is finding the closest token in Doc2 for each token in Doc1.
         * Then do a weighted sum of the distances found by the respective weights of tokens in Doc1.
         *
         * @param nbow1 L1-Normalized BOW representation of Doc1 (sorted by token index)
         * @param nbow2 L1-Normalized BOW representation of Doc2 (sorted by token index)
         * @return RWMD distance value
         */
        DistanceValue computeDistance(const Document& nbow1, const Document& nbow2) {
            // For a faster execution and tighter WMD approximation, documents do not share tokens.
            // E.g: If nbow1 and nbow2 contain the token 'AI' with nbow1['AI'] = 0.1 and nbow2['AI'] = 0.2.
            //      We take the diff so that 'AI' is removed from nbow1 and nbow2['AI'] = 0.1.
            //      This is the same as transporting weight = 0.1 with cost 0.
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);

            // Compute distance matrix between embeddings in nbow1 and nbow2
            std::vector<TokenIndex> tokens1 = C_Util::getNbowIndices(diffNbow1), tokens2 = C_Util::getNbowIndices(diffNbow2);
            EigenDistanceMatrix D = m_embeddings->computeDistances(tokens1, tokens2);

            // Compute WMD relaxation - ignoring one constraint at a time
            DistanceValue l12 = computeAsymmetricDistance(diffNbow1, diffNbow2, D, false);
            DistanceValue l21 = computeAsymmetricDistance(diffNbow2, diffNbow1, D, true);
            return std::max(l12, l21);
        }

    private:

        std::shared_ptr<C_Embeddings> m_embeddings;

        /**
         * Compute the Asymmetrical Relaxed Word Mover's Distance (RWMD) from Doc1 to Doc2.
         * The idea is finding the closest token in Doc2 for each token in Doc1.
         * Then do a weighted sum of the distances found by the respective weights of tokens in Doc1.
         *
         * @param nbow1 L1-Normalized BOW representation of Doc1 (sorted by token index)
         * @param nbow2 L1-Normalized BOW representation of Doc2 (sorted by token index)
         * @param D Distance matrix from all tokens in Doc1 to the tokens in Doc2
         * @param isTransposed Whether to invert the indices being accessed in the matrix D
         * @return Asymmetric RWMD distance value
         */
        DistanceValue computeAsymmetricDistance(const Document& nbow1, const Document& nbow2,
                                                const EigenDistanceMatrix& D, bool isTransposed) {
            DistanceValue value = 0;
            for(std::size_t i = 0; i < nbow1.size(); i++) {
                TokenWeight weight = nbow1[i].second;
                DistanceValue minDistance = std::numeric_limits<DistanceValue>::max();
                for(std::size_t j = 0; j < nbow2.size(); j++) {
                    DistanceValue distance = isTransposed? D.coeffRef(j, i) : D.coeffRef(i, j);
                    if(nbow1[i].first == nbow2[j].first) {
                        weight = std::max(0.0f, weight - nbow2[j].second);
                    } else if(distance < minDistance) {
                        minDistance = distance;
                    }
                }
                value += weight * minDistance;
            }
            return value;
        }

    };

}

#endif //FAST_WMD_C_RWMD_H
