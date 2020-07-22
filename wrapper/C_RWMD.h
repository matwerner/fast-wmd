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

        float computeDistance(const std::vector<std::pair<unsigned int, float>>& nbow1,
                              const std::vector<std::pair<unsigned int, float>>& nbow2) {
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);
            std::vector<unsigned int> tokens1 = C_Util::getNbowIndices(diffNbow1), tokens2 = C_Util::getNbowIndices(diffNbow2);
            Eigen::MatrixXf D = m_embeddings->computeDistances(tokens1, tokens2);
            float l12 = computeAsymmetricDistance(diffNbow1, diffNbow2, D, false);
            float l21 = computeAsymmetricDistance(diffNbow2, diffNbow1, D, true);
            return std::max(l12, l21);
        }

    private:

        std::shared_ptr<C_Embeddings> m_embeddings;

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
