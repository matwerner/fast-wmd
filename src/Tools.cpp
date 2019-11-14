#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>

#include "Tools.hpp"

/**
 * Get documents triplets.
 */
std::vector<std::tuple<int, int, int>> Tools::getTriplets(const std::string& fileName) {
	std::ifstream inFile (fileName);
	if (!inFile.is_open()) {
		throw std::runtime_error("Unable to open file. Path: " + fileName);
	}

	std::string line;
	std::vector<std::tuple<int, int, int>> triplets;
	while (std::getline(inFile, line)) {
		std::stringstream inputStream(line);
		std::istream_iterator<std::string> begin(inputStream);
		std::istream_iterator<std::string> end;
		std::vector<std::string> inputVector(begin, end);

		if(inputVector.size() != 3) {
			throw std::runtime_error("Unable to parse line. Line: " + line + "Path: " + fileName);
		}

		int doc0 = std::stoi(inputVector[0]);
		int doc1 = std::stoi(inputVector[1]);
		int doc2 = std::stoi(inputVector[2]);
		triplets.emplace_back(doc0, doc1, doc2);
	}
	inFile.close();

	return triplets;
}

/**
 * Get triplets documents.
 */
Eigen::SparseMatrix<int> Tools::getTripletsDocuments(const std::string& fileName, int tokensSize) {
	std::ifstream inFile (fileName);
	if (!inFile.is_open()) {
		throw std::runtime_error("Unable to open file. Path: " + fileName);
	}

	std::string line;
	std::vector<int> labels;
	std::vector<std::vector<int>> texts;
	while (std::getline(inFile, line)) {
		std::stringstream inputStream(line);
		std::istream_iterator<std::string> begin(inputStream);
		std::istream_iterator<std::string> end;
		std::vector<std::string> inputVector(begin, end);

		std::vector<int> v(inputVector.size());
		for(int i = 0; i < inputVector.size(); i++) {
			v[i] = std::stoi(inputVector[i]);
		}

		texts.push_back(v);
	}
	inFile.close();

	std::vector<Eigen::Triplet<int>> tripletList;
	for(int i = 0; i < texts.size(); i++) {
		Eigen::VectorXf v = Eigen::VectorXf::Zero(tokensSize);
		for(int tokenId: texts[i]) {
			v[tokenId] += 1;
		}
		for(int j = 0; j < tokensSize; j++) {
			if(v[j] == 0) {
				continue;
			}
			tripletList.emplace_back(j, i, Tools::MASS_MULT * v[j] / texts[i].size());
		}
	}
	Eigen::SparseMatrix<int> mat(tokensSize, texts.size());
	mat.setFromTriplets(tripletList.begin(), tripletList.end());

	// Normalizing BOW vectors
	for(std::size_t i = 0; i < texts.size(); i++) {
		if(texts[i].empty()) {
			continue;
		}
		mat.coeffRef(texts[i][0], i) -= (mat.col(i).sum() - Tools::MASS_MULT);
	}
	return mat;
}

/**
 * Get kusner documents.
 */
std::pair<std::vector<int>, Eigen::SparseMatrix<int>> Tools::getDocuments(const std::string& fileName, int tokensSize) {
	std::ifstream inFile (fileName);
	if (!inFile.is_open()) {
		throw std::runtime_error("Unable to open file. Path: " + fileName);
	}

	std::string line;
	std::vector<int> labels;
	std::vector<std::vector<int>> texts;
	while (std::getline(inFile, line)) {
		std::stringstream inputStream(line);
		std::istream_iterator<std::string> begin(inputStream);
		std::istream_iterator<std::string> end;
		std::vector<std::string> inputVector(begin, end);

		std::vector<int> v(inputVector.size() - 1);
		for(int i = 1; i < inputVector.size(); i++) {
			v[i-1] = std::stoi(inputVector[i]);
		}

		labels.push_back(std::stoi(inputVector[0]));
		texts.push_back(v);
	}
	inFile.close();

	std::vector<Eigen::Triplet<int>> tripletList;
	for(int i = 0; i < texts.size(); i++) {
		Eigen::VectorXf v = Eigen::VectorXf::Zero(tokensSize);
		for(int tokenId: texts[i]) {
			v[tokenId] += 1;
		}
		for(int j = 0; j < tokensSize; j++) {
			if(v[j] == 0) {
				continue;
			}
			tripletList.emplace_back(j, i, Tools::MASS_MULT * v[j] / texts[i].size());
		}
	}
	Eigen::SparseMatrix<int> mat(tokensSize, texts.size());
	mat.setFromTriplets(tripletList.begin(), tripletList.end());

	// Normalizing BOW vectors
	for(std::size_t i = 0; i < texts.size(); i++) {
		mat.coeffRef(texts[i][0], i) -= (mat.col(i).sum() - Tools::MASS_MULT);
	}
	return std::pair<std::vector<int>, Eigen::SparseMatrix<int>>(labels, mat);
}

/**
 * Get embeddings.
 */
Eigen::MatrixXf Tools::getEmbeddings(const std::string& fileName) {
	std::ifstream inFile (fileName);
	if (!inFile.is_open()) {
		throw std::runtime_error("Unable to open file. Path: " + fileName);
	}

	int tokenSize, embeddingSize;
	inFile >> tokenSize >> embeddingSize;
	inFile.ignore(); // ignore '\n'

	std::string line;
	Eigen::MatrixXf embeddings(embeddingSize, tokenSize);
	int i;
	for(i = 0; i < tokenSize || inFile.peek() != std::ifstream::traits_type::eof(); i++) {
		std::string lineError = "Line: " + std::to_string(i) + ". File: " + fileName + "\n";

		std::getline(inFile, line);
		std::stringstream inputStream(line);
		std::istream_iterator<std::string> begin(inputStream);
		std::istream_iterator<std::string> end;
		std::vector<std::string> inputVector(begin, end);

		if(inputVector.size() != embeddingSize + 1) {
			throw std::runtime_error("Embedding size is " + std::to_string(inputVector.size() - 1) +
									 ", but it should be " + std::to_string(embeddingSize) + ".\n" + lineError);
		}

		int tokenId = std::stoi(inputVector[0]);
		for(int j = 0; j < embeddingSize; j++) {
			try {
				embeddings(j,tokenId) = std::stof(inputVector[j+1]);
			}
			catch (const std::exception& e) {
				throw std::runtime_error("Embedding value could not be parsed.\n" + lineError);
			}
		}
	}
	inFile.close();

	if(i < tokenSize) {
		throw std::runtime_error("Number of tokens is incorrect. File: " + fileName + "\n");
	}
	else if(inFile.peek() != std::ifstream::traits_type::eof()) {
		throw std::runtime_error("Number of tokens is incorrect. File: " + fileName + "\n");
	}
	return embeddings;
}

/**
 * Compute Related Words cache.
 */
std::pair<std::vector<std::unordered_map<int, int>>, int> Tools::computeRelatedWordsCacheFromEmbeddings(const Eigen::MatrixXf& embeddings, int r, bool verbose) {
	int tokensSize = (int) embeddings.cols();

	Eigen::VectorXf embeddingsSquareNorm(tokensSize);
	for(int i = 0; i < tokensSize; i++) {
		embeddingsSquareNorm[i] = embeddings.col(i).squaredNorm();
	}

	std::vector<std::unordered_map<int, int>> cacheDistancesMap(tokensSize, std::unordered_map<int, int>());
	for(auto& cacheDistancesDict: cacheDistancesMap) {
		cacheDistancesDict.reserve(r);
	}

	double invalidDistancesTotal = 0.0;
	std::size_t invalidDistancesCount = 0;
	for(size_t i = 0; i < tokensSize; i++) {
		if(verbose && i % 1000 == 0) {
			std::cout << "Processed " << i << " of " << tokensSize << " tokens" << std::endl;
		}

		Eigen::VectorXf embedding = embeddings.col(i);
		float embeddingSquareNorm = embeddingsSquareNorm[i];

		// Compute norms
		Eigen::VectorXf SS = embeddingsSquareNorm;
		Eigen::VectorXf TT = embeddingSquareNorm * Eigen::VectorXf::Ones(tokensSize);
		Eigen::VectorXf ST = 2 * embeddings.transpose() * embedding;
		Eigen::VectorXf DSquared = SS - ST + TT;

		// Fix negative "zeros"
		for(int j = 0; j < DSquared.size(); j++) {
			if(DSquared.coeff(j) < 0.0f) DSquared.coeffRef(j) = 0.0f;
		}

		Eigen::VectorXf D = DSquared.cwiseSqrt();

		// Get the correct threshold to be used
		float threshold = std::numeric_limits<float>::max();
		if (r < tokensSize) {
			std::vector<float> distances(D.data(), D.data() + D.size());
			std::nth_element(distances.begin(), distances.begin() + r, distances.end());
			threshold = distances[r] + 0.000001f;
		}

		for(size_t j = 0; j < tokensSize; j++) {
			if(i == j) continue;

			float distance = D.coeff(j);
			if(distance < threshold) {
				cacheDistancesMap[i][j] = (int) std::round(Tools::COST_MULT * distance);
				cacheDistancesMap[j][i] = (int) std::round(Tools::COST_MULT * distance);
			}
			else {
				invalidDistancesTotal += distance;
				invalidDistancesCount++;
			}
		}
	}
	int invalidDistanceMean = (int) std::round(COST_MULT * invalidDistancesTotal / invalidDistancesCount);
	return std::pair<std::vector<std::unordered_map<int, int>>, int>(cacheDistancesMap, invalidDistanceMean);
}

/**
 * Compute related words from clusters
 */
std::pair<std::vector<std::unordered_map<int, int>>, int> Tools::computeRelatedWordsCacheFromClusters(const std::vector<std::vector<int>>& clusters,
																									  const Eigen::MatrixXf& embeddings, int r, bool verbose) {
	std::vector<std::unordered_map<int, int>> cacheDistancesMap(embeddings.cols());
	std::size_t invalidDistancesTotal = 0, invalidDistancesCount = 0;
	for(const std::vector<int>& cluster: clusters) {
		if(cluster.empty()) continue;

		Eigen::MatrixXf clusterEmbeddings(embeddings.rows(), cluster.size());
		for(int i = 0; i < cluster.size(); i++) {
			int idx = cluster[i];
			clusterEmbeddings.col(i) = embeddings.col(idx);
		}

		std::pair<std::vector<std::unordered_map<int, int>>, int> clusterRelatedWords = Tools::computeRelatedWordsCacheFromEmbeddings(clusterEmbeddings, r, verbose);
		for(int i = 0; i < cluster.size(); i++) {
			int tokenIdx = cluster[i];
			std::size_t numRelatedWords = clusterRelatedWords.first[i].size();

			cacheDistancesMap[tokenIdx].reserve(numRelatedWords);
			for(const auto& relatedToken: clusterRelatedWords.first[i]) {
				int relatedTokenIdx = cluster[relatedToken.first];
				cacheDistancesMap[tokenIdx][relatedTokenIdx] = relatedToken.second;
			}
			invalidDistancesTotal += (cluster.size() - numRelatedWords - 1) * clusterRelatedWords.second;
			invalidDistancesCount += (cluster.size() - numRelatedWords - 1);
		}
	}
	int invalidDistanceMean = (int) std::round(invalidDistancesTotal / invalidDistancesCount);
	return std::make_pair(std::move(cacheDistancesMap), invalidDistanceMean);
};

/**
 * Compute Euclidean distance between docs embeddings.
 */
Eigen::MatrixXi Tools::computeEuclideanDistanceBetweenDocsEmbeddings(const Eigen::MatrixXf& doc1Embeddings,
																	 const Eigen::MatrixXf& doc2Embeddings) {
	int doc1Size = doc1Embeddings.cols(), doc2Size = doc2Embeddings.cols();

	Eigen::MatrixXf SS = Eigen::MatrixXf::Zero(doc1Size, doc2Size).colwise()
						 + doc1Embeddings.colwise().norm().transpose();
	Eigen::MatrixXf TT = Eigen::MatrixXf::Zero(doc1Size, doc2Size).array().rowwise()
						 + doc2Embeddings.colwise().norm().array();
	Eigen::MatrixXf ST = 2 * (doc1Embeddings.transpose() * doc2Embeddings);
	Eigen::MatrixXf DSquared = SS - ST + TT;

	// Fix negative "zeros"
	for(int j = 0; j < DSquared.cols(); j++) {
		for(Eigen::MatrixXf::InnerIterator it(DSquared, j); it; ++it) {
			if(it.value() < 0.0f) DSquared.coeffRef(it.row(), j) = 0.0f;
		}
	}

	return (Tools::COST_MULT * DSquared.cwiseSqrt()).cast<int>();
}

/**
 * Compute doc cache.
 */
Eigen::MatrixXi Tools::computeDocCache(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings) {
	Eigen::MatrixXf docEmbeddings(embeddings.rows(), doc.nonZeros());
	int i = 0;
	for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
		docEmbeddings.col(i) = embeddings.col(it.row());
		i++;
	}
	return Tools::computeEuclideanDistanceBetweenDocsEmbeddings(docEmbeddings, embeddings);
}

/**
 * Compute the nearest word in a document to each word in vocabulary.
 */
Eigen::VectorXi Tools::computeNearestDistancesToDoc(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings) {
	// Compute distance from each word of document to each word of vocabulary
	Eigen::MatrixXi D = Tools::computeDocCache(doc, embeddings);

	// Get the closest word to each word in vocabulary
	Eigen::VectorXi nearDists = D.colwise().minCoeff();

	// In case of the words contained in current doc get second closest word
	for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
		Eigen::VectorXi v = D.col(it.row());
		int j = 0, minDistance = std::numeric_limits<int>::max();
		for (Eigen::SparseVector<int>::InnerIterator it2(doc); it2; ++it2) {
			if(it.row() != it2.row()) {
				minDistance = std::min(minDistance, v[j]);
			}
			j++;
		}
		nearDists.coeffRef(it.row()) = minDistance;
	}
	return nearDists;
}

/**
 * Compute the nearest word in each document to each word in vocabulary.
 */
Eigen::MatrixXi Tools::computeNearestDistancesToDocs(const Eigen::SparseMatrix<int>& docs, const Eigen::MatrixXf& embeddings) {
	// Initialize sparse matrix containing, for each document, the nearest distance to each word in vocabulary
	Eigen::MatrixXi nearestDistancesToDocsCache(embeddings.cols(), docs.cols());

	// For each document, find the closest word in document to each word in vocabulary
	for(int i = 0; i < docs.cols(); i++) {
		Eigen::SparseVector<int> doc = docs.col(i);
		nearestDistancesToDocsCache.col(i) = Tools::computeNearestDistancesToDoc(doc, embeddings);
	}

	return nearestDistancesToDocsCache;
}

/**
 * Compute the nearest related word in a document to each word in vocabulary.
 */
std::unordered_map<int, int> Tools::computeNearestRelatedDistancesToDoc(const Eigen::SparseVector<int>& doc,
		                                                                const std::vector<std::unordered_map<int, int>>& relatedWordsDict) {
	// Find minimum distance by word of the vocabulary
	std::unordered_map<int, int> minDistanceToWordMap;
	for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {

		// For each related word to word in document, update whether is the lowest distance
		for(const auto& it2: relatedWordsDict[it.row()]) {
			int wordIdx = it2.first, distance = it2.second;
			const auto entry = minDistanceToWordMap.find(wordIdx);
			if(entry == minDistanceToWordMap.end() || entry->second > distance) {
				minDistanceToWordMap[wordIdx] = distance;
			}
		}
	}

	return minDistanceToWordMap;
}

/**
 * Compute the nearest related word in each document to each word in vocabulary.
 */
std::vector<std::unordered_map<int, int>> Tools::computeNearestRelatedDistancesToDocs(const Eigen::SparseMatrix<int>& docs,
                                                                                      const std::vector<std::unordered_map<int, int>>& relatedWordsDict) {
	// Initialize sparse matrix containing, for each document, the nearest distance to each word in vocabulary
	std::vector<std::unordered_map<int, int>> nearestDistancesToDocsCache(docs.cols());

	// For each document, find the closest word in document to each word in vocabulary
	for(int i = 0; i < docs.cols(); i++) {
		Eigen::SparseVector<int> doc = docs.col(i);
		nearestDistancesToDocsCache[i] = Tools::computeNearestRelatedDistancesToDoc(doc, relatedWordsDict);
	}

	return nearestDistancesToDocsCache;
}

/**
 * Compute predicted label.
 */
int Tools::computePredictedLabel(std::vector<std::pair<float, int>>& distancePairs, int numLabels, int k,
								 bool isOrdered) {
	if(!isOrdered) {
		std::partial_sort(distancePairs.begin(), distancePairs.begin() + k, distancePairs.end());
	}
	if(k <= 1) {
		return distancePairs[0].second;
	}
	std::vector<size_t> nearestLabels(numLabels, 0);
	for(int i = 0; i < k; i++) {
		nearestLabels[distancePairs[i].second]++;
	}
	int max = (int) *std::max_element(nearestLabels.begin(), nearestLabels.end());

	int predictedLabel = -1;
	for(int i = 0; i < numLabels; i++) {
		if(nearestLabels[i] < max) {
			continue;
		}
		else if(predictedLabel < 0) {
			predictedLabel = i;
		}
		else {
			predictedLabel = Tools::computePredictedLabel(distancePairs, numLabels, k/2, true);
			break;
		}
	}
	return predictedLabel;
}

/**
 * Compute Error Rate.
 */
float Tools::computeErrorRate(const std::vector<int>& predictedLabels, const std::vector<int>& trueLabels) {
	int truePositive = 0;
	for(int i = 0; i < predictedLabels.size(); i++) {
		truePositive += predictedLabels[i] == trueLabels[i]? 1 : 0;
	}
	return 1.0f - (truePositive / (float) predictedLabels.size());
}
