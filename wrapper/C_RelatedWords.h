#ifndef FAST_WMD_C_RELATEDWORDS_H
#define FAST_WMD_C_RELATEDWORDS_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "C_Embeddings.h"

namespace fastwmd {

    class C_RelatedWords {

    public:

        C_RelatedWords() {}

        C_RelatedWords(const std::shared_ptr<C_Embeddings>& embeddings, unsigned int r) {
            unsigned int numEmbeddings = embeddings->getNumEmbeddings();

            // Reserve expected memory space
            m_r = r;
            m_cache = std::vector<std::unordered_map<unsigned int, float>>(numEmbeddings, std::unordered_map<unsigned int, float>());
            for(auto& dict: m_cache) {
                dict.reserve(m_r);
            }

            // Pre-compute
            const Eigen::MatrixXf& embeddingsMatrix = embeddings->getEmbeddings();
            Eigen::VectorXf embeddingsSquareNorm = embeddingsMatrix.colwise().squaredNorm();
            Eigen::VectorXf SS = embeddingsSquareNorm;

            double unrelatedDistancesTotal = 0.0;
            unsigned int unrelatedDistancesCount = 0;
            for(unsigned int i = 0; i < numEmbeddings; i++) {
                // Compute norms
                Eigen::VectorXf TT = embeddingsSquareNorm[i]* Eigen::VectorXf::Ones(numEmbeddings);
                Eigen::VectorXf ST = 2 * embeddingsMatrix.transpose() * embeddingsMatrix.col(i);
                Eigen::VectorXf D = (SS - ST + TT).cwiseSqrt();

                // Get the correct threshold to be used
                // The word itself is not considered part of the r-th closest words
                float threshold = std::numeric_limits<float>::max();
                if (r < numEmbeddings) {
                    std::vector<float> distances(D.data(), D.data() + D.size());
                    std::nth_element(distances.begin(), distances.begin() + m_r, distances.end());
                    threshold = distances[m_r] + std::numeric_limits<float>::epsilon();
                }

                for(unsigned int j = 0; j < numEmbeddings; j++) {
                    if(i == j) {
                        continue;
                    } else if(D.coeff(j) < threshold) {
                        m_cache[i][j] = D.coeff(j);
                        m_cache[j][i] = D.coeff(j);
                    } else {
                        unrelatedDistancesTotal += D.coeff(j);
                        unrelatedDistancesCount++;
                    }
                }
            }
            m_maximumDistance = (float) (unrelatedDistancesTotal / unrelatedDistancesCount);
        }

        unsigned int getR() {
            return m_r;
        }

        const std::unordered_map<unsigned int, float>& getRelatedWords(unsigned int index) {
            return m_cache[index];
        }

        float getMaximumDistance() {
            return m_maximumDistance;
        }

    private:

        unsigned int m_r;
        std::vector<std::unordered_map<unsigned int, float>> m_cache;
        float m_maximumDistance;

    };
}

#endif //FAST_WMD_C_RELATEDWORDS_H
