#ifndef FAST_WMD_TOOLS_HPP
#define FAST_WMD_TOOLS_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <utility>
#include <vector>

class Tools {

public:

	/**
	 * Load the triplets dataset. Each line correspond to a triplet in format: <doc-id-0> <doc-id-1> <doc-id-2>.
	 * Doc0 and Doc1 are considered related while Doc2 is considered unrelated.
	 * @param fileName Path to the triplets dataset.
	 * @return Vector
	 */
	static std::vector<std::tuple<int, int, int>> getTriplets(const std::string& fileName);

	/**
	 * Load the triplets documents. Parse the documents file to its BoW representation.
	 * Each line correspond to a document and its format must be: <token-id-0> <token-id-1> ... <token-id-D>
	 * @param fileName Path to the documents dataset.
	 * @param tokensSize Number of distinct tokens (Needed for creating BoW vector).
	 * @return Bow representation of each document in t_fileName
	 */
	static Eigen::SparseMatrix<int> getTripletsDocuments(const std::string& fileName, int tokensSize);

	/**
	 * Load the document dataset. Parse the documents file to its BoW representation.
	 * Each line correspond to a document and its format must be: <label> <token-id-0> <token-id-1> ... <token-id-D>
	 * @param fileName Path to the documents dataset.
	 * @param tokensSize Number of distinct tokens (Needed for creating BoW vector).
	 * @return Label and Bow representation of each document in t_fileName
	 */
	static std::pair<std::vector<int>, Eigen::SparseMatrix<int>> getDocuments(const std::string& fileName, int tokensSize);

	/**
	 * Load all embeddings contained in file.
	 * First line format must be: <vocabulary-size> <embedding-size>
	 * After each line corresponds to a word embedding and its format must be: <token-id> <dim-0> <dim-1> ... <dim-d>
	 * @param fileName Path to all embeddings being used by the documents.
	 * @return Embeddings matrix. Each column represents to one word embedding.
	 */
	static Eigen::MatrixXf getEmbeddings(const std::string& fileName);

	/**
	 * Compute the r-th closest words to each word in embeddings. Also return the Maximum distance composed by the mean
	 * of all excluded distances.
	 * Distances are return as integers so they can be used with OR-Tools algorithms.
	 * @param embeddings Embeddings Matrix.
	 * @param r Number of related word for each word.
	 * @param verbose Dump info (Debug purposes).
	 * @return
	 */
	static std::pair<std::vector<std::unordered_map<int, int>>, int> computeRelatedWordsCacheFromEmbeddings(const Eigen::MatrixXf& embeddings, int r, bool verbose=false);

	/**
	 * Compute the r-th closest words to each word in embeddings given the word embedding clusters.
	 * Also return the Maximum distance composed by the mean of all excluded distances.
	 * Distances are return as integers so they can be used with OR-Tools algorithms.
	 * @param clusters Word Embedding clusters.
	 * @param embeddings Embeddings Matrix.
	 * @param r Number of related word for each word.
	 * @param verbose Dump info (Debug purposes).
	 * @return
	 */
	static std::pair<std::vector<std::unordered_map<int, int>>, int> computeRelatedWordsCacheFromClusters(const std::vector<std::vector<int>>& clusters,
																										  const Eigen::MatrixXf& embeddings, int r, bool verbose);

	/**
	 * Compute the euclidean distance between the embeddings of a documents pair.
	 * Distances are return as integers so they can be used with OR-Tools algorithms.
	 * @param doc1Embeddings Document 1 embeddings.
	 * @param doc2Embeddings Document 2 embeddings.
	 * @return Distance Eigen Matrix between all embeddings from document 1 and 2.
	 */
	static Eigen::MatrixXi computeEuclideanDistanceBetweenDocsEmbeddings(const Eigen::MatrixXf& doc1Embeddings,
																		 const Eigen::MatrixXf& doc2Embeddings);

	/**
	 * Compute the euclidean distance between the Word Embeddings embeddings in the document and the vocabulary.
	 * @param doc Document representation.
	 * @param embeddings All Word Embeddings contained in the vocabulary.
	 * @return Distance Eigen Matrix between all embeddings from vocabulary and the document.
	 */
	static Eigen::MatrixXi computeDocCache(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings);

    /**
     * Compute the nearest word in a document to each word in vocabulary.
     * @param doc Document representation of a document in dataset.
     * @param embeddings All Word Embeddings contained in the vocabulary.
     * @return
     */
    static Eigen::VectorXi computeNearestDistancesToDoc(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings);

	/**
	 * Compute the nearest word in each document to each word in vocabulary.
	 * @param docs Document representation of each document in dataset.
	 * @param embeddings All Word Embeddings contained in the vocabulary.
	 * @return
	 */
	static Eigen::MatrixXi computeNearestDistancesToDocs(const Eigen::SparseMatrix<int>& docs, const Eigen::MatrixXf& embeddings);

    /**
     * Compute the nearest related word in document to each word in vocabulary.
     * @param doc Document representation of a document in dataset.
     * @param relatedWordsDict
     * @return
     */
    static std::unordered_map<int, int> computeNearestRelatedDistancesToDoc(const Eigen::SparseVector<int>& doc,
                                                                            const std::vector<std::unordered_map<int, int>>& relatedWordsDict);

	/**
	 * Compute the nearest related word in each document to each word in vocabulary.
	 * @param docs Document representation of each document in dataset.
	 * @param relatedWordsDict
	 * @return
	 */
	static std::vector<std::unordered_map<int, int>> computeNearestRelatedDistancesToDocs(const Eigen::SparseMatrix<int>& docs,
	                                                                                      const std::vector<std::unordered_map<int, int>>& relatedWordsDict);
	/**
	 * Compute the predicted label of the k-NN. In case of ties, k is divided by 2 until there are no more ties.
	 * @param distancePairs Distance of all train documents to the document being classified and their respective Labels.
	 * @param numLabels Number of labels (must be sequential 0 to X)
	 * @param k Number of neighbours in k-NN.
	 * @param isOrdered whether the distancePairs vector is ordered.
	 * @return Predicted label of the document being classified.
	 */
	static int computePredictedLabel(std::vector<std::pair<float, int>>& distancePairs, int numLabels, int k,
									 bool isOrdered=false);

	/**
	 * Compute the error rate from the prediction of the k-NN.
	 * @param predictedLabels Predicted labels for all test documents using k-NN.
	 * @param trueLabels True labels of all test documents.
	 * @return Prediction error rate.
	 */
	static float computeErrorRate(const std::vector<int>& predictedLabels, const std::vector<int>& trueLabels);

private:

	const static int MASS_MULT = 1000 * 1000;   ///< weights quantization constant
	const static int COST_MULT = 1000;		 	///< costs quantization constant
};


#endif //FAST_WMD_TOOLS_HPP
