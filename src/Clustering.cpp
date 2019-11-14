#include "Clustering.hpp"

#include <ctime>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

/**
 * Compute K-Means using Light Weight Coresets.
 */
std::vector<int> Clustering::computeLightweightCoresets(const Eigen::MatrixXf& vectors, int numClusters, int coresetSize,
        int numIterations) {
    std::pair<Eigen::MatrixXf, std::vector<float>> pair = getCoreset(vectors, coresetSize);
    std::pair<Eigen::MatrixXf, std::vector<int>> result = computeWeightedKMeans(pair.first, pair.second, numClusters, numIterations);
    std::vector<int> assignments = assignCentroidsToVectors(vectors, result.first);
    return assignments;
}

/**
 * Get the coreset.
 */
std::pair<Eigen::MatrixXf, std::vector<float>> Clustering::getCoreset(const Eigen::MatrixXf& vectors, int coresetSize) {
    int vectorSize = vectors.rows();
    int numVectors = vectors.cols();

    Eigen::VectorXf meanVector = vectors.rowwise().mean();

    Eigen::VectorXf SS = vectors.colwise().norm();
    Eigen::VectorXf TT = Eigen::VectorXf::Ones(numVectors) * meanVector.norm();
    Eigen::VectorXf ST = 2 * (vectors.transpose() * meanVector);
    Eigen::VectorXf D = (SS - ST + TT);

    double cumDist = D.sum();
    std::vector<float> q(numVectors, 0.0f);
    for(int i = 0; i < numVectors; i++) {
        q[i] = 0.5f * (1.0f / numVectors) + 0.5f * D(i) / cumDist;
    }
    std::default_random_engine generator;
    std::discrete_distribution<> custom_dist(q.begin(), q.end());

    Eigen::MatrixXf coreset(vectorSize, coresetSize);
    std::vector<float> weights(coresetSize);
    for(int i = 0; i < coresetSize; i++) {
        int vectorIdx = custom_dist(generator);
        coreset.col(i) = vectors.col(vectorIdx);
        weights[i] = 1.0f / (coresetSize * q[i]);
    }

    return  std::make_pair(coreset, weights);
}

/**
 * Initialize centroids.
 */
Eigen::MatrixXf Clustering::initializeCentroids(const Eigen::MatrixXf& vectors, int numClusters) {
    int vectorSize = vectors.rows();
    int numVectors = vectors.cols();

    // Initialize clusters
    std::default_random_engine generator;
    std::uniform_int_distribution<int>  uniform_dist(0, numVectors - 1);

    Eigen::MatrixXf centroids = Eigen::MatrixXf::Zero(vectorSize, numClusters);
    int clusterIdx = uniform_dist(generator);

    centroids.col(0) = vectors.col(clusterIdx);

    double cumSquareDistances = 0.0;
    std::vector<double> squareDistances(numVectors, std::numeric_limits<float>::max());

    Eigen::VectorXf SS = vectors.colwise().norm();
    for(int i = 1; i < numClusters; i++) {
        Eigen::VectorXf TT = Eigen::VectorXf::Ones(numVectors) * centroids.col(i - 1).norm();
        Eigen::VectorXf ST = 2 * (vectors.transpose() * centroids.col(i - 1));
        Eigen::VectorXf D = (SS - ST + TT);

        for(int j = 0; j < numVectors; j++) {
            if(D(j) > squareDistances[j]) {
                cumSquareDistances += D(j);
                cumSquareDistances -= squareDistances[j];
                squareDistances[j] = D(j);
            }
        }

        std::discrete_distribution<> custom_dist(squareDistances.begin(), squareDistances.end());
        clusterIdx = custom_dist(generator);
        centroids.col(i) = vectors.col(clusterIdx);
    }

    return centroids;
}

/**
 * Compute the Weighted K-Means.
 */
std::pair<Eigen::MatrixXf, std::vector<int>> Clustering::computeWeightedKMeans(const Eigen::MatrixXf& vectors,
        std::vector<float> weights, int numClusters, int numIterations) {
    Eigen::MatrixXf centroids = initializeCentroids(vectors, numClusters);
    std::vector<int> assignments = assignCentroidsToVectors(vectors, centroids);
    for(int i = 0; i < numIterations; i++) {
        centroids = recomputeCentroids(vectors, weights, assignments, numClusters);
        std::vector<int> newAssignments = assignCentroidsToVectors(vectors, centroids);
        if(assignments == newAssignments) {
            return std::make_pair(std::move(centroids), assignments);
        }
        assignments = newAssignments;
    }
    return std::make_pair(std::move(centroids), assignments);
}

/**
 * Assign a cluster to each vector.
 */
std::vector<int> Clustering::assignCentroidsToVectors(const Eigen::MatrixXf& vectors, Eigen::MatrixXf& centroids) {
    int numVectors = vectors.cols();
    int numCentroids = centroids.cols();

    Eigen::MatrixXf SS = Eigen::MatrixXf::Zero(numVectors, numCentroids).colwise()
                         + vectors.colwise().norm().transpose();
    Eigen::MatrixXf TT = Eigen::MatrixXf::Zero(numVectors, numCentroids).array().rowwise()
                         + centroids.colwise().norm().array();
    Eigen::MatrixXf ST = 2 * (vectors.transpose() * centroids);
    Eigen::MatrixXf D = (SS - ST + TT);

    std::vector<int> assignments(numVectors, -1);
    for(int i = 0; i < numVectors; i++) {
        int clusterIdx = -1;
        D.row(i).minCoeff(&clusterIdx);
        assignments[i] = clusterIdx;
    }

    return assignments;
}

/**
 * Recompute the centroids.
 */
Eigen::MatrixXf Clustering::recomputeCentroids(const Eigen::MatrixXf& vectors, std::vector<float> weights,
        std::vector<int> assignments, int numClusters) {
    int vectorSize = vectors.rows();
    int numVectors = vectors.cols();

    Eigen::MatrixXf centroids = Eigen::MatrixXf::Zero(vectorSize, numClusters);

    std::vector<float> totalWeightPerCentroid(numClusters, 0);
    for(int i = 0; i < numVectors; i++) {
        int clusterIdx = assignments[i];
        centroids.col(clusterIdx) += (weights[i] * vectors.col(i));
        totalWeightPerCentroid[clusterIdx] += weights[i];
    }

    for(int i = 0; i < numClusters; i++) {
        centroids.col(i) /= totalWeightPerCentroid[i];
    }

    return centroids;
}
