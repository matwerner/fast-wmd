#ifndef FAST_WMD_C_DISTANCE_H
#define FAST_WMD_C_DISTANCE_H

#include <vector>
#include "C_CONSTANTS.h"

namespace fastwmd {

    class C_Distance {

    public:

        C_Distance() {}

        /**
         * Compute the distance from Doc1 and Doc2.
         *
         * @param nbow1 Data structure containing the tokens in Document 1 and their respective weights
         * @param nbow2 Data structure containing the tokens in Document 2 and their respective weights
         * @return Distance value
         */
        virtual DistanceValue computeDistance(const Document& nbow1, const Document& nbow2) = 0;

        /**
         * Compute the distances from the docs in collection 1 to the docs in collection 2.
         *
         * @param nbow1 Data structure vector containing tokens and their respective weights of collection 1
         * @param nbow2 Data structure vector containing tokens and their respective weights of collection 2
         * @return Distance values between documents in collections 1 and 2
         */
        std::vector<std::vector<DistanceValue>> computeDistances(const std::vector<Document>& nbows1, const std::vector<Document>& nbows2) {
            std::vector<std::vector<DistanceValue>> distances(nbows1.size(), std::vector<DistanceValue>(nbows2.size()));
            for(std::size_t i = 0; i < nbows1.size(); i++) {
                for(std::size_t j = 0; j < nbows2.size(); j++) {
                    distances[i][j] = computeDistance(nbows1[i], nbows2[j]);
                }
            }
            return distances;
        }

        /**
         * Compute the distances between all docs in the given collection.
         *
         * @param nbows Data structure vector containing tokens and their respective weights of a collection
         * @return Distance values between documents in collection
         */
        std::vector<std::vector<DistanceValue>> computeDistances(const std::vector<Document>& nbows) {
            std::vector<std::vector<DistanceValue>> distances(nbows.size(), std::vector<DistanceValue>(nbows.size()));
            for(std::size_t i = 0; i < nbows.size()-1; i++) {
                for(std::size_t j = i+1; j < nbows.size(); j++) {
                    DistanceValue distance = computeDistance(nbows[i], nbows[j]);
                    distances[i][j] = distance;
                    distances[j][i] = distance;
                }
            }
            return distances;
        }

    };
}


#endif //FAST_WMD_C_DISTANCE_H
