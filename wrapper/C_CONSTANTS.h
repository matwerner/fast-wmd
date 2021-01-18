#ifndef FAST_WMD_C_CONSTANTS_H
#define FAST_WMD_C_CONSTANTS_H

#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace fastwmd {

// Document-related representations
typedef unsigned int TokenIndex;
typedef float TokenWeight;
typedef std::vector<std::pair<TokenIndex, TokenWeight>> Document;
typedef std::unordered_map<TokenIndex, TokenWeight> HashedDocument;

// Embedding-related representations
typedef float EmbeddingWeight;
typedef Eigen::MatrixXf EigenEmbeddingMatrix;
typedef Eigen::VectorXf EigenEmbeddingVector;

// Distance-related representions
typedef float DistanceValue;
typedef std::unordered_map<TokenIndex, DistanceValue> HashedRelatedWords;
typedef Eigen::MatrixXf EigenDistanceMatrix;
typedef Eigen::VectorXf EigenDistanceVector;

}

#endif //FAST_WMD_C_CONSTANTS_H