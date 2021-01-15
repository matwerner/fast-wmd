#ifndef FAST_WMD_C_RWMD_H
#define FAST_WMD_C_RWMD_H

#include <memory>
#include "C_Distance.h"
#include "C_Embeddings.h"
#include "C_Util.h"

namespace fastwmd {

    class C_RWMD: public C_Distance<std::vector<std::pair<unsigned int, float>>> {

    public:

        C_RWMD() {}

        C_RWMD(const std::shared_ptr<C_Embeddings>& embeddings): m_embeddings(embeddings) {}

        /**
         * Compute the Relaxed Word Mover's Distance (RWMD) from document 1 to document 2.
         * The idea is finding the closest token in D2 for each token in D1.
         * Then do a weighted sum of the distances found and the respective weights of tokens in D1.
         *
         * @param nbow1 List of tokens in Document 1 and their respective weights
         * @param nbow2 List of tokens in Document 2 and their respective weights
         * @return RWMD distance
         */
        float computeDistance(const std::vector<std::pair<unsigned int, float>>& nbow1,
                              const std::vector<std::pair<unsigned int, float>>& nbow2) {
            // For a faster execution and tighter WMD approximation, documents do not share tokens.
            // E.g: If nbow1 and nbow2 contain the token 'AI' with nbow1['AI'] = 0.1 and nbow2['AI'] = 0.2.
            //      We take the diff so that 'AI' is removed from nbow1 and nbow2['AI'] = 0.1.
            //      This is the same of transporting weight = 0.1 with cost 0.
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);

            // Compute distance matrix between embeddings in nbow1 and nbow2
            std::vector<unsigned int> tokens1 = C_Util::getNbowIndices(diffNbow1), tokens2 = C_Util::getNbowIndices(diffNbow2);
            Eigen::MatrixXf D = m_embeddings->computeDistances(tokens1, tokens2);

            // Compute WMD relaxation - ignoring one constraint at a time
            float l12 = computeAsymmetricDistance(diffNbow1, diffNbow2, D, false);
            float l21 = computeAsymmetricDistance(diffNbow2, diffNbow1, D, true);
            return std::max(l12, l21);
        }

    private:

        std::shared_ptr<C_Embeddings> m_embeddings;

        /**
         * Compute the Asymmetrical Relaxed Word Mover's Distance (RWMD) from document 1 to document 2.
         * The idea is finding the closest token in D2 for each token in D1.
         * Then do a weighted sum of the distances found and the respective weights of tokens in D1.
         *
         * @param nbow1 List of tokens in Document 1 and their respective weights
         * @param nbow2 List of tokens in Document 2 and their respective weights
         * @param D Distance between all tokens in document 1 to the tokens in document 2
         * @param isTransposed Whether to invert the indices been accessed in the matrix D
         * @return Asymmetric RWMD distance
         */
        float computeAsymmetricDistance(const std::vector<std::pair<unsigned int, float>>& nbow1,
                                        const std::vector<std::pair<unsigned int, float>>& nbow2,
                                        const Eigen::MatrixXf& D, bool isTransposed) {
            float value = 0;
            for(unsigned int i = 0; i < nbow1.size(); i++) {
                float weight = nbow1[i].second, minDistance = std::numeric_limits<float>::max();
                for(unsigned int j = 0; j < nbow2.size(); j++) {
                    float distance = isTransposed? D.coeffRef(j, i) : D.coeffRef(i, j);
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
