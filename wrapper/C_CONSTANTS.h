#ifndef FAST_WMD_C_CONSTANTS_H
#define FAST_WMD_C_CONSTANTS_H

#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace fastwmd {

typedef unsigned int TokenIndex;
typedef float TokenWeight;
typedef float DistanceValue;
typedef std::vector<std::pair<TokenIndex, TokenWeight>> Document;
typedef std::unordered_map<TokenIndex, TokenWeight> HashedDocument;
typedef std::unordered_map<TokenIndex, DistanceValue> HashedRelatedWords;
typedef Eigen::MatrixXf EigenDistanceMatrix;
typedef Eigen::VectorXf EigenDistanceVector;

}

#endif //FAST_WMD_C_CONSTANTS_H