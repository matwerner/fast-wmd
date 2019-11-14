#ifndef FAST_WMD_CLUSTERING_H
#define FAST_WMD_CLUSTERING_H

#include <eigen3/Eigen/Dense>
#include <vector>

class Clustering {

public:

    /**
     * Compute K-Means using Light Weight Coresets.
     *
     * @param vectors Vectors to be clustered
     * @param numClusters Number of clusters
     * @param coresetSize Size of the coreset
     * @param numIterations Maximum number of iterations allowed during k-means computation
     * @return
     */
    static std::vector<int> computeLightweightCoresets(const Eigen::MatrixXf& vectors, int numClusters, int coresetSize,
            int numIterations);

    /**
     * Compute the Weighted K-Means.
     *
     * @param vectors Vectors to be clustered
     * @param weights 'Importance' given to each vector
     * @param numClusters Number of clusters
     * @param numIterations Maximum number of iterations allowed during K-Means computation
     * @return
     */
    static std::pair<Eigen::MatrixXf, std::vector<int>> computeWeightedKMeans(const Eigen::MatrixXf& vectors,
            std::vector<float> weights, int numClusters, int numIterations);

private:

    /**
     * Get the coreset of the vectors passed as parameters.
     *
     * @param vectors Vectors to be clustered
     * @param coresetSize Size of the coreset
     * @return Pair containg the vectors selected to represent the coreset and their respective weights
     */
    static std::pair<Eigen::MatrixXf, std::vector<float>> getCoreset(const Eigen::MatrixXf& vectors, int coresetSize);

    /**
    * Generate cluster's centroids using K-Means++ strategy.

    * @param vectors Vectors to be clustered
    * @param numClusters Number of clusters
    * @return Initial vector centroids
    */
    static Eigen::MatrixXf initializeCentroids(const Eigen::MatrixXf& vectors, int numClusters);

    /**
     * Assign a cluster to each vector.
     *
     * @param vectors Vectors to be clustered
     * @param centroids Centroid vectors
     * @return vector containing the assigned clusters to each vector
     */
    static std::vector<int> assignCentroidsToVectors(const Eigen::MatrixXf& vectors, Eigen::MatrixXf& centroids);

    /**
     * Compute the new centroids given the current cluster assignment.
     *
     * @param vectors Vectors to be clustered
     * @param weights 'Importance' given to each vector
     * @param assignments Current cluster assignment of all vectors
     * @param numClusters Number of clusters
     * @return The new centroid vectors
     */
    static Eigen::MatrixXf recomputeCentroids(const Eigen::MatrixXf& vectors, std::vector<float> weights,
            std::vector<int> assignments, int numClusters);

};


#endif //FAST_WMD_CLUSTERING_H
