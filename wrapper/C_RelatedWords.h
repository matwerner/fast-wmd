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

        /**
         * Compute and store the r-closest word embeddings to each word embedding
         * Also compute a distance representing the distance between any word embedding pair not explicitly stored
         * 
         * @param embeddings Word embedding data structure
         * @param r Number of closest word to store
         */
        C_RelatedWords(const std::shared_ptr<C_Embeddings>& embeddings, std::size_t r) {
            std::size_t numEmbeddings = embeddings->getNumEmbeddings();

            // Reserve expected memory space
            m_r = r;
            m_cache = std::vector<HashedRelatedWords>(numEmbeddings, HashedRelatedWords());
            for(auto& dict: m_cache) {
                dict.reserve(m_r);
            }

            // Pre-compute
            const EigenEmbeddingMatrix& embeddingsMatrix = embeddings->getEmbeddings();
            EigenDistanceVector embeddingsSquareNorm = embeddingsMatrix.colwise().squaredNorm();
            EigenDistanceVector SS = embeddingsSquareNorm;

            double unrelatedDistancesTotal = 0.0;
            std::size_t unrelatedDistancesCount = 0;
            for(std::size_t i = 0; i < numEmbeddings; i++) {
                // Compute norms
                EigenDistanceVector TT = embeddingsSquareNorm[i]* EigenDistanceVector::Ones(numEmbeddings);
                EigenDistanceVector ST = 2 * embeddingsMatrix.transpose() * embeddingsMatrix.col(i);
                EigenDistanceVector D = (SS - ST + TT).cwiseSqrt();

                // Get the correct threshold to be used
                // The word itself is not considered part of the r-th closest words
                DistanceValue threshold = std::numeric_limits<DistanceValue>::max();
                if (r < numEmbeddings) {
                    std::vector<DistanceValue> distances(D.data(), D.data() + D.size());
                    std::nth_element(distances.begin(), distances.begin() + m_r, distances.end());
                    threshold = distances[m_r] + std::numeric_limits<DistanceValue>::epsilon();
                }

                for(std::size_t j = 0; j < numEmbeddings; j++) {
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
            m_maximumDistance = (DistanceValue) (unrelatedDistancesTotal / unrelatedDistancesCount);
        }

        /**
         * Get the number of closest tokens stored per token
         * 
         * @return Number of closest tokens stored per token
         */
        std::size_t getR() {
            return m_r;
        }

        /**
         * Get the r-closest tokens to the given token index
         * 
         * @param index Token index
         * @return r-closest tokens and their respective distances to token index
         */
        const HashedRelatedWords& getRelatedWords(TokenIndex index) {
            return m_cache[index];
        }

        /**
         * Get the distance value representing the distance between any token pair not explicitly stored
         * 
         * @return Maximum distance between any token pair
         */
        DistanceValue getMaximumDistance() {
            return m_maximumDistance;
        }

    private:

        std::size_t m_r;
        std::vector<HashedRelatedWords> m_cache;
        DistanceValue m_maximumDistance;

    };
}

#endif //FAST_WMD_C_RELATEDWORDS_H
