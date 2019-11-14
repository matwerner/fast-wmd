#ifndef FAST_WMD_DISTANCES_H
#define FAST_WMD_DISTANCES_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>

class Distances {

public:

    /**
     * Cosine distance implementation.
     *
     * @param doc1
     * @param doc2
     * @return
     */
    static float computeCosineDistance(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2);

    /**
     * Euclidean distance implementation.
     *
     * @param doc1
     * @param doc2
     * @return
     */
    static float computeEuclideanDistance(const Eigen::VectorXi &doc1, const Eigen::VectorXi &doc2);

    /**
     * Word Mover's Distance implementation.
     *
     * @param doc1
     * @param doc2
     * @param embeddings
     * @return
     */
    static float computeWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2,
                            const Eigen::MatrixXf& embeddings);

    /**
     * Relaxed Word Mover's Distance implementation.
     *
     * @param doc1
     * @param doc2
     * @param embeddings
     * @return
     */
    static float computeRWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2,
                             const Eigen::MatrixXf& embeddings);

    /**
     * Linear Complexity Word Mover's Distance implementation.
     *
     * @param doc1
     * @param nearestToDoc1
     * @param doc2
     * @param nearestToDoc2
     * @return
     */
    static float computeLinearRWMD(const Eigen::SparseVector<int> &doc1, const Eigen::VectorXi &nearDistancesToDoc1,
                                   const Eigen::SparseVector<int> &doc2, const Eigen::VectorXi &nearDistancesToDoc2);

    /**
     * Word Centroid Distance implementation for one document.
     *
     * @param docs
     * @param embeddings
     * @return
     */
    static Eigen::VectorXi computeWCD(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings);

    /**
     * Word Centroid Distance implementation for multiple documents.
     *
     * @param docs
     * @param embeddings
     * @return
     */
    static Eigen::MatrixXi computeWCD(const Eigen::SparseMatrix<int>& docs, const Eigen::MatrixXf& embeddings);

    /**
     * Word Mover's Distance + Edges reduction implementation.
     * Following formulation proposed by Pele et al. ("Fast and robust Earth Mover's Distances").
     * @param doc1
     * @param doc2
     * @param r
     * @param relatedWordsCache
     * @return
     */
    static float computeRelWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2, int r,
                               const std::pair<std::vector<std::unordered_map<int, int>>, int>& relatedWordsCache);

    /**
     * Relaxed Word Mover's Distance + Edges reduction implementation.
     *
     * @param doc1
     * @param doc2
     * @param doc1Hash
     * @param doc2Hash
     * @param relatedWordsCache
     * @return
     */
    static float computeRelRWMD(const Eigen::SparseVector<int> &doc1, const std::unordered_map<int, int> &doc1Hash,
                                const Eigen::SparseVector<int> &doc2, const std::unordered_map<int, int> &doc2Hash,
                                const std::pair<std::vector<std::unordered_map<int, int>>, int>& relatedWordsCache);

    /**
     * Linear Complexity Word Mover's Distance + Edges reduction implementation.
     *
     * @param doc1
     * @param nearestToDoc1
     * @param doc2
     * @param nearestToDoc2
     * @return
     */
    static float computeLinearRelatedRWMD(const Eigen::SparseVector<int> &doc1, const std::unordered_map<int, int> &nearDistancesToDoc1,
                                          const Eigen::SparseVector<int> &doc2, const std::unordered_map<int, int> &nearDistancesToDoc2,
                                          int maximumDistance);
};


#endif //FAST_WMD_DISTANCES_H
