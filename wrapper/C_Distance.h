#ifndef FAST_WMD_C_DISTANCE_H
#define FAST_WMD_C_DISTANCE_H

#include <vector>

namespace fastwmd {

    template <typename T>
    class C_Distance {

    public:

        C_Distance() {}

        virtual float computeDistance(const T& nbow1, const T& nbow2) = 0;

        std::vector<std::vector<float>> computeDistances(const std::vector<T>& nbows1, const std::vector<T>& nbows2) {
            std::vector<std::vector<float>> distances(nbows1.size(), std::vector<float>(nbows2.size()));
            for(size_t i = 0; i < nbows1.size(); i++) {
                for(size_t j = 0; j < nbows2.size(); j++) {
                    distances[i][j] = computeDistance(nbows1[i], nbows2[j]);
                }
            }
            return distances;
        }

        std::vector<std::vector<float>> computeDistances(const std::vector<T>& nbows) {
            std::vector<std::vector<float>> distances(nbows.size(), std::vector<float>(nbows.size()));
            for(size_t i = 0; i < nbows.size()-1; i++) {
                for(size_t j = i+1; j < nbows.size(); j++) {
                    float distance = computeDistance(nbows[i], nbows[j]);
                    distances[i][j] = distance;
                    distances[j][i] = distance;
                }
            }
            return distances;
        }

    };
}


#endif //FAST_WMD_C_DISTANCE_H
