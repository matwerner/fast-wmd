#include <ctime>
#include <iostream>

#include "Clustering.hpp"
#include "Distances.hpp"
#include "Tools.hpp"

const static std::vector<int> R_PARAMETERS = {1, 2, 4, 8, 16, 32, 64, 128};

std::tuple<float, float, float> runKNN(const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& trainDataset,
									   const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& testDataset,
									   const Eigen::MatrixXf& embeddings, int r, int k,
									   const std::string& functionName, bool verbose) {
    // Some distances are required to use Dictionaries instead of Sparse vectors due to O(1) access.
    // For simplicity, we instantiate the dictionaries here, so we do not need to also pass them as parameters.
    // Because of this, we do not take this step into account during the preprocessing phase.
    std::vector<std::unordered_map<int, int>> testHashes, trainHashes;

    // For now, only rel-rwmd needs hash data structure - Avoid instantiating unnecessary data
    if(functionName == "rel-rwmd") {
        testHashes.resize(testDataset.first.size());
        for (int i = 0; i < testHashes.size(); i++) {
            Eigen::SparseVector<int> doc = testDataset.second.col(i);
            testHashes[i].reserve(doc.nonZeros());
            for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
                testHashes[i][it.row()] = it.value();
            }
        }

        trainHashes.resize(trainDataset.first.size());
        for (int i = 0; i < trainHashes.size(); i++) {
            Eigen::SparseVector<int> doc = trainDataset.second.col(i);
            trainHashes[i].reserve(doc.nonZeros());
            for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
                trainHashes[i][it.row()] = it.value();
            }
        }
    }

    // Preprocess related words if REL-WMD / REL-RWMD / LC-REL-RWMD was selected
    std::time_t preStart = std::clock();
    std::pair<std::vector<std::unordered_map<int, int>>, int> relatedWordsCache;
    if(functionName == "rel-wmd" || functionName == "rel-rwmd" || functionName == "lc-rel-rwmd") {
        relatedWordsCache = Tools::computeRelatedWordsCacheFromEmbeddings(embeddings, r);
    }

    // Preprocess auxiliary data structures
    Eigen::MatrixXi trainCentroids, trainNearDistancesToDocs;
    std::vector<std::unordered_map<int, int>> trainNearRelatedDistancesToDocs;
    // Get Word Centroid Distance representations
    if(functionName == "wcd") {
        trainCentroids = Distances::computeWCD(trainDataset.second, embeddings);
    }
    // Get nearest distances from each word in vocabulary to document
    else if(functionName == "lc-rwmd") {
        trainNearDistancesToDocs = Tools::computeNearestDistancesToDocs(trainDataset.second, embeddings);
    }
    // Get nearest related distances from each word in vocabulary to document
    else if(functionName == "lc-rel-rwmd") {
        trainNearRelatedDistancesToDocs = Tools::computeNearestRelatedDistancesToDocs(trainDataset.second, relatedWordsCache.first);
    }

    float preTime = (std::clock() - preStart) / (float) CLOCKS_PER_SEC;

    // Get number of labels - must be sequential 0 to X
    int numLabels = 1 + (*std::max_element(trainDataset.first.begin(), trainDataset.first.end()));

    // Run KNN
    std::time_t knnStart = std::clock();
    std::vector<int> predictedLabels(testDataset.first.size()), predictedLabels2(testDataset.first.size());
    std::vector<std::pair<float, int>> distancePairs(trainDataset.first.size()), distancePairs2(trainDataset.first.size());
    Eigen::SparseVector<int> doc1, doc2;
    Eigen::MatrixXi doc1Cache;
    Eigen::VectorXi centroid1, centroid2, nearDistancesToDoc1;
	std::unordered_map<int, int> nearRelatedDistancesToDoc1, doc1Hash;
    for(int i = 0; i < testDataset.first.size(); i++) {
        doc1 = testDataset.second.col(i);
        if(functionName == "wcd") {
            centroid1 = Distances::computeWCD(doc1, embeddings);
        } else if(functionName == "rel-rwmd") {
            doc1Hash = testHashes[i];
        } else if(functionName == "lc-rwmd") {
            nearDistancesToDoc1 = Tools::computeNearestDistancesToDoc(doc1, embeddings);
        } else if(functionName == "lc-rel-rwmd") {
            nearRelatedDistancesToDoc1 = Tools::computeNearestRelatedDistancesToDoc(doc1, relatedWordsCache.first);
        }

        for(int j = 0; j < trainDataset.first.size(); j++) {
            if(functionName == "wcd") {
                centroid2 = trainCentroids.col(j);
            } else {
                doc2 = trainDataset.second.col(j);
            }

            std::time_t funcStart = std::clock();
            if(functionName == "cosine")              distancePairs[j].first = Distances::computeCosineDistance(doc1, doc2);
            else if(functionName == "wcd")            distancePairs[j].first = Distances::computeEuclideanDistance(centroid1, centroid2);
            else if(functionName == "wmd")            distancePairs[j].first = Distances::computeWMD(doc1, doc2, embeddings);
            else if(functionName == "rwmd")           distancePairs[j].first = Distances::computeRWMD(doc1, doc2, embeddings);
            else if(functionName == "lc-rwmd")        distancePairs[j].first = Distances::computeLinearRWMD(doc1, nearDistancesToDoc1,
                                                                                                            doc2, trainNearDistancesToDocs.col(j));
            else if(functionName == "rel-wmd")        distancePairs[j].first = Distances::computeRelWMD(doc1, doc2, r, relatedWordsCache);
            else if(functionName == "rel-rwmd")       distancePairs[j].first = Distances::computeRelRWMD(doc1, doc1Hash, doc2, trainHashes[j], relatedWordsCache);
            else if(functionName == "lc-rel-rwmd")    distancePairs[j].first = Distances::computeLinearRelatedRWMD(doc1, nearRelatedDistancesToDoc1,
                                                                                                                   doc2, trainNearRelatedDistancesToDocs[j],
                                                                                                                   relatedWordsCache.second);
            else throw std::invalid_argument("Unknown function: " + functionName);
            distancePairs[j].second = trainDataset.first[j];

            float funcTime = (std::clock() - funcStart) / (float) CLOCKS_PER_SEC;
            if(verbose) {
                std::cout << doc1.nonZeros() << "\t" << doc2.nonZeros() << "\t" << funcTime << "\t" << distancePairs[j].first  << std::endl;
            }
        }
        predictedLabels[i] = Tools::computePredictedLabel(distancePairs, numLabels, k);
    }
    float knnTime = (std::clock() - knnStart) / (float) CLOCKS_PER_SEC;

    // Get error rate
    float errorRate = Tools::computeErrorRate(predictedLabels, testDataset.first);

    return std::make_tuple(errorRate, preTime, knnTime);
}

int selectBestRForKusner(const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& dataset,
						 const Eigen::MatrixXf& embeddings, int k, const std::string& functionName, bool verbose) {
    // Check if r is needed
    if(functionName != "rel-wmd" && functionName != "rel-rwmd" && functionName != "lc-rel-rwmd") {
        return -1;
    }

    std::size_t numDocs = dataset.first.size();
    std::vector<int> indices(numDocs, 0);
    for(int i = 0; i < numDocs; i++) {
        indices[i] = i;
    }

    // Get random documents permutation - Fixed seed for reproducibility
    // std::shuffle(indices.begin(), indices.end(), std::default_random_engine(0));
    std::random_shuffle(indices.begin(), indices.end());

    std::size_t numPartitions = 5;
    std::size_t numDocsPerPartition = (std::size_t) std::ceil((numDocs + 1.0f) / numPartitions);

    // Initialize partitions
    std::vector<std::vector<int>> partitions(numPartitions);
    for(int i = 0; i < numPartitions; i++) {
        partitions[i].reserve(numDocsPerPartition);
    }

    // Assign each document index to its partition
    for(int i = 0; i < numDocs; i++) {
        std::size_t idx = i / numDocsPerPartition;
        partitions[idx].push_back(indices[i]);
    }

    // Initialize error rate for each r tested
    std::vector<float> testErrors(R_PARAMETERS.size(), 0.0f);
    for(std::size_t i = 0; i < numPartitions; i++) {
        std::size_t testSize = partitions[i].size();
        std::size_t trainSize = numDocs - testSize;

        std::vector<int> trainLabels(trainSize, -1), testLabels(testSize, -1);
        std::vector<Eigen::Triplet<int>> trainTriplets, testTriplets;

        std::size_t trainIdx = 0, testIdx = 0;
        for(std::size_t j = 0; j < numPartitions; j++) {
            if(i == j) {
                for(int docIdx: partitions[j]) {
                    for (Eigen::SparseMatrix<int>::InnerIterator it(dataset.second, docIdx); it; ++it) {
                        testTriplets.emplace_back(it.row(), testIdx, it.value());
                    }
                    testLabels[testIdx] = dataset.first[docIdx];
                    testIdx++;
                }
            } else {
                for(int docIdx: partitions[j]) {
                    for (Eigen::SparseMatrix<int>::InnerIterator it(dataset.second, docIdx); it; ++it) {
                        trainTriplets.emplace_back(it.row(), trainIdx, it.value());
                    }
                    trainLabels[trainIdx] = dataset.first[docIdx];
                    trainIdx++;
                }
            }
        }

        Eigen::SparseMatrix<int> trainDocuments(dataset.second.rows(), trainSize), testDocuments(dataset.second.rows(), testSize);
        trainDocuments.setFromTriplets(trainTriplets.begin(), trainTriplets.end());
        testDocuments.setFromTriplets(testTriplets.begin(), testTriplets.end());

        std::pair<std::vector<int>, Eigen::SparseMatrix<int>> trainDataset = std::make_pair(trainLabels, trainDocuments);
        std::pair<std::vector<int>, Eigen::SparseMatrix<int>> testDataset = std::make_pair(testLabels, testDocuments);
        for(std::size_t j = 0; j < R_PARAMETERS.size(); j++) {
            int r = R_PARAMETERS[j];
            std::tuple<float, float, float> result = runKNN(trainDataset, testDataset, embeddings, r, k, functionName, verbose);
            testErrors[j] += std::get<0>(result);
			std::cout << i << "\t" << r << "\t" << std::get<0>(result) << "\t" <<  std::get<1>(result) << "\t" <<  std::get<2>(result) << std::endl;
        }
    }

    // Get the smaller (faster) r that is within 1% of the smallest error rate found
    float minErrorRate = *std::min_element(testErrors.begin(), testErrors.end());
    for(std::size_t j = 0; j < R_PARAMETERS.size(); j++) {
        if(testErrors[j] < 1.01 * minErrorRate) {
            return R_PARAMETERS[j];
        }
    }
    return -1;
}

/**
 * Run kusner experiment.
 *
 * @param trainDatasetFilePath Path to the documents composing the Train dataset.
 * @param testDatasetFilePath Path to the documents composing the Test dataset.
 * @param embeddingsFilePath Path to all embeddings being used by the documents.
 * @param k Number of Neighbours in k-NN.
 * @param functionName Distance function to be used with k-NN.
 * @param r Number of related word for each word.
 * @param verbose Whether to dump info for each document pair being processed.
 */
void runKusnerExperiment(const std::string& trainDatasetFilePath, const std::string& testDatasetFilePath,
						 const std::string& embeddingsFilePath, int k, const std::string& functionName, int r, bool verbose) {
	Eigen::MatrixXf embeddings = Tools::getEmbeddings(embeddingsFilePath);
	std::pair<std::vector<int>, Eigen::SparseMatrix<int>> trainDataset = Tools::getDocuments(trainDatasetFilePath, embeddings.cols());
	std::pair<std::vector<int>, Eigen::SparseMatrix<int>> testDataset = Tools::getDocuments(testDatasetFilePath, embeddings.cols());

	if(r < 1) {
		r = selectBestRForKusner(trainDataset, embeddings, k, functionName, verbose);
	}

    std::tuple<float, float, float> result = runKNN(trainDataset, testDataset, embeddings, r, k, functionName, verbose);
    float errorRate = std::get<0>(result);
    float preTime = std::get<1>(result);
    float knnTime = std::get<2>(result);

	// Dump results to Console
	std::string SEP = "\t";
	std::string filesString = trainDatasetFilePath + SEP + testDatasetFilePath + SEP + embeddingsFilePath + SEP;
	std::string paramsString = std::to_string(k) + SEP + functionName + SEP + std::to_string(r) + SEP;
	std::string timesString = std::to_string(preTime) + SEP + std::to_string(knnTime) + SEP;
	std::cout <<  filesString << paramsString << timesString << errorRate << std::endl;
}

std::tuple<float, float, float> runRelatedDocumentPairsIdentification(const std::vector<std::tuple<int, int, int>>& tripletsDataset,
																	  const Eigen::SparseMatrix<int>& documents, const Eigen::MatrixXf& embeddings,
																	  int numClusters, int maxIterations,
																	  const std::string& functionName, int r, bool verbose) {
    // Some distances are required to use Dictionaries instead of Sparse vectors due to O(1) access.
    // For simplicity, we instantiate the dictionaries here, so we do not need to also pass them as parameters.
    // Because of this, we do not take this step into account during the preprocessing phase.
    std::vector<std::unordered_map<int, int>> docHashes;

    // For now, only rel-rwmd needs hash data structure - Avoid instantiating unnecessary data
    if(functionName == "rel-rwmd") {
        docHashes.resize(documents.cols());
        for(int i = 0; i < docHashes.size(); i++) {
            Eigen::SparseVector<int> doc = documents.col(i);
            docHashes[i].reserve(doc.nonZeros());
            for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
                docHashes[i][it.row()] = it.value();
            }
        }
    }

	// Preprocess related words if REL-WMD / REL-RWMD / LC-REL-RWMD was selected
	std::time_t preStart = std::clock();
	std::pair<std::vector<std::unordered_map<int, int>>, int> relatedWordsCache;
	Eigen::MatrixXi centroids, nearDistancesToDocs;
	std::vector<std::unordered_map<int, int>> nearRelatedDistancesToDocs;
	if(functionName == "rel-wmd" || functionName == "rel-rwmd" || functionName == "lc-rel-rwmd") {
		if(numClusters > 0) {
			if (verbose) {
				std::cout << "Computing related words given clusters..." << std::endl;
			}
            std::pair<Eigen::MatrixXf, std::vector<int>> result = Clustering::computeWeightedKMeans(embeddings,
                    std::vector<float>(embeddings.cols(), 1.0f), numClusters, maxIterations);
			std::vector<std::vector<int>> embeddingClusters(numClusters, std::vector<int>());
			for(int i = 0; i < result.second.size(); i++) {
                embeddingClusters[result.second[i]].push_back(i);
			}
			relatedWordsCache = Tools::computeRelatedWordsCacheFromClusters(embeddingClusters, embeddings, r, verbose);
		} else {
			if(verbose) {
				std::cout << "Computing related words..." << std::endl;
			}
			relatedWordsCache = Tools::computeRelatedWordsCacheFromEmbeddings(embeddings, r, verbose);
		}

		if(functionName == "lc-rel-rwmd") {
			if(verbose) {
				std::cout << "Computing nearest related distances to each doc..." << std::endl;
			}
            nearRelatedDistancesToDocs = Tools::computeNearestRelatedDistancesToDocs(documents, relatedWordsCache.first);
		}
	} else if(functionName == "wcd") {
		centroids = Distances::computeWCD(documents, embeddings);
	} else if(functionName == "lc-rwmd") {
        nearDistancesToDocs = Tools::computeNearestDistancesToDocs(documents, embeddings);
	}

	float preTime = (std::clock() - preStart) / (float) CLOCKS_PER_SEC;
	if(verbose) {
		std::cout << preTime << std::endl;
	}

	// Run experiment
	int numCorrect = 0;
	std::time_t experimentStart = std::clock();
	Eigen::SparseVector<int> doc1, doc2, doc3, otherDoc;
	Eigen::VectorXi centroid1, centroid2, centroid3, otherCentroid, nearDistancesToDoc1;
	std::unordered_map<int, int> nearRelatedDistancesToDoc1;
	for(const std::tuple<int, int, int>& triplet: tripletsDataset) {
	    int idx1 = std::get<0>(triplet), idx2 = std::get<1>(triplet), idx3 = std::get<2>(triplet), otherIdx;
		if(functionName == "wcd") {
			centroid1 = centroids.col(idx1);
			centroid2 = centroids.col(idx2);
			centroid3 = centroids.col(idx3);
		} else {
			doc1 = documents.col(idx1);
			doc2 = documents.col(idx2);
			doc3 = documents.col(idx3);
			if(functionName == "lc-rwmd") {
				nearDistancesToDoc1 = nearDistancesToDocs.col(idx1);
			} else if(functionName == "lc-rel-rwmd") {
				nearRelatedDistancesToDoc1 = nearRelatedDistancesToDocs[idx1];
			}
		}

		if(functionName != "wcd" && (doc1.nonZeros() == 0 || doc2.nonZeros() == 0 || doc3.nonZeros() == 0)) {
			continue;
		}

		float tripletDistances[2];
		for(int j = 0; j < 2; j++) {
            otherIdx = j == 0? idx2 : idx3;
			if(functionName == "wcd") {
				otherCentroid = j == 0? centroid2 : centroid3;
			} else {
				otherDoc = j == 0? doc2 : doc3;
			}

			std::time_t funcStart = std::clock();
			if(functionName == "cosine")              tripletDistances[j] = Distances::computeCosineDistance(doc1, otherDoc);
			else if(functionName == "wcd")            tripletDistances[j] = Distances::computeEuclideanDistance(centroid1, otherCentroid);
			else if(functionName == "wmd")            tripletDistances[j] = Distances::computeWMD(doc1, otherDoc, embeddings);
			else if(functionName == "rwmd")           tripletDistances[j] = Distances::computeRWMD(doc1, otherDoc, embeddings);
            else if(functionName == "lc-rwmd")        tripletDistances[j] = Distances::computeLinearRWMD(doc1, nearDistancesToDoc1,
                                                                                                         otherDoc, nearDistancesToDocs.col(otherIdx));
			else if(functionName == "rel-wmd")        tripletDistances[j] = Distances::computeRelWMD(doc1, otherDoc, r, relatedWordsCache);
			else if(functionName == "rel-rwmd")       tripletDistances[j] = Distances::computeRelRWMD(doc1, docHashes[idx1],
			                                                                                          otherDoc, docHashes[otherIdx],
			                                                                                          relatedWordsCache);
			else if(functionName == "lc-rel-rwmd")    tripletDistances[j] = Distances::computeLinearRelatedRWMD(doc1, nearRelatedDistancesToDoc1,
																												otherDoc, nearRelatedDistancesToDocs[otherIdx],
																												relatedWordsCache.second);
			else throw std::invalid_argument("Unknown function: " + functionName);

			float funcTime = (std::clock() - funcStart) / (float) CLOCKS_PER_SEC;
			if(verbose) {
				std::cout << doc1.nonZeros() << "\t" << otherDoc.nonZeros() << "\t" << funcTime << std::endl;
			}
		}

		numCorrect += (tripletDistances[0] < tripletDistances[1]? 1 : 0);
	}
	float experimentTime = (std::clock() - experimentStart) / (float) CLOCKS_PER_SEC;

	// Get error rate
	float accuracy = numCorrect / (float) tripletsDataset.size();

	// For some motive before ending function there is a peak in memory use
	relatedWordsCache.first.clear();
	return std::make_tuple(accuracy, preTime, experimentTime);
}

/**
 * Run triplets experiment.
 *
 * @param tripletsFilePath Path to the triplets the dataset.
 * @param documentsFilePath Path to all documents in the triplets dataset.
 * @param embeddingsFilePath Path to all embeddings being used by the documents.
 * @param functionName Distance function to measure similarity among the triplets.
 * @param r Number of related word for each word.
 * @param verbose Whether to dump info for each document pair being processed.
 */
void runTripletsExperiment(const std::string& tripletsFilePath, const std::string& documentsFilePath,
						   const std::string& embeddingsFilePath, int numClusters, int maxIterations,
						   const std::string& functionName, int r, bool verbose) {

	if(verbose) {
		std::cout << "Reading embeddings..." << std::endl;
	}
	Eigen::MatrixXf embeddings = Tools::getEmbeddings(embeddingsFilePath);

    if(verbose) {
        std::cout << "Reading triplets dataset..." << std::endl;
    }
    std::vector<std::tuple<int, int, int>> tripletsDataset = Tools::getTriplets(tripletsFilePath);

    if(verbose) {
        std::cout << "Reading documents..." << std::endl;
    }
    Eigen::SparseMatrix<int> documents = Tools::getTripletsDocuments(documentsFilePath, embeddings.cols());

	// Run experiment
	std::tuple<float, float, float> result =
			runRelatedDocumentPairsIdentification(tripletsDataset, documents, embeddings, numClusters, maxIterations, functionName, r, verbose);
	float accuracy = std::get<0>(result);
	float preTime = std::get<1>(result);
	float experimentTime = std::get<2>(result);

	// Dump results to Console
	std::string SEP = "\t";
	std::string filesString = tripletsFilePath + SEP + documentsFilePath+ SEP + embeddingsFilePath + SEP;
	std::string paramsString = functionName + SEP + std::to_string(r) + SEP;
	std::string timesString = std::to_string(preTime) + SEP + std::to_string(experimentTime) + SEP;
	std::cout <<  filesString << paramsString << timesString << accuracy << std::endl;
}


int main(int argc, const char * argv[]) {
	std::string trainDatasetFilePath, testDatasetFilePath, tripletsFilePath, documentsFilePath, embeddingsFilePath, functionName;
    int k = -1, r = -1, numClusters = -1, maxIterations = -1;
	bool verbose = false;

	std::string mode = argv[1];
	if(mode != "kusner" && mode != "triplets") {
		throw std::invalid_argument("Mode must 'kusner' or 'triplets'. Unknown mode: " + mode);
	}

	// Parse cmd parameters
	for(int i = 2; i < argc; i+=2) {
		std::string cmd = argv[i];
		if(cmd == "--tr") {
			trainDatasetFilePath = argv[i+1];
		}
		else if(cmd == "--te") {
			testDatasetFilePath = argv[i+1];
		}
		else if(cmd == "--trip") {
			tripletsFilePath = argv[i+1];
		}
		else if(cmd == "--docs") {
			documentsFilePath = argv[i+1];
		}
		else if(cmd == "--emb") {
			embeddingsFilePath = argv[i+1];
		}
		else if(cmd == "--func") {
			functionName = argv[i+1];
		}
		else if(cmd == "--k") {
			k = std::stoi(argv[i+1]);
		}
		else if(cmd == "--r") {
			r = std::stoi(argv[i+1]);
		}
        else if(cmd == "--num_clusters") {
            numClusters = std::stoi(argv[i+1]);
        }
        else if(cmd == "--max_iter") {
            maxIterations = std::stoi(argv[i+1]);
        }
		else if(cmd == "--verbose") {
			verbose = (argv[i+1] == std::string("true"));
		}
		else if(cmd == "--help") {
			if(mode == "kusner") {
				std::cout << "Usage: <exe> kusner [options]" << std::endl << std::endl;
				std::cout << "Input Options:" << std::endl << std::endl;
				std::cout << "--tr <filepath>:   " << "Train dataset filepath"      << std::endl << std::endl;
				std::cout << "--te <filepath>:   " << "Test dataset filepath"       << std::endl << std::endl;
				std::cout << "--emb <filepath>:  " << "Embeddings filepath"         << std::endl << std::endl;
				std::cout << "--func <function>: " << "Function to be used"         << std::endl << std::endl;
				std::cout << "--k X:             " << "Number of neighbours in kNN" << std::endl << std::endl;
				std::cout << "--r X:             " << "Number of related words"     << std::endl << std::endl;
			} else {
				std::cout << "Usage: <exe> triplets [options]" << std::endl << std::endl;
				std::cout << "Input Options:" << std::endl << std::endl;
				std::cout << "--trip <filepath>: " << "Triplets dataset filepath"   << std::endl << std::endl;
				std::cout << "--docs <filepath>: " << "Documents filepath"          << std::endl << std::endl;
				std::cout << "--emb <filepath>:  " << "Embeddings filepath"         << std::endl << std::endl;
				std::cout << "--num_clusters X:  " << "Number of clusters"          << std::endl << std::endl;
				std::cout << "--max_iter X:      " << "Maximum number of iterations during clustering"  << std::endl << std::endl;
				std::cout << "--func <function>: " << "Function to be used"         << std::endl << std::endl;
				std::cout << "--r X:             " << "Number of related words"     << std::endl << std::endl;
			}

			return 0;
		}
		else {
			throw std::invalid_argument("Unknown option: " + cmd);
		}
	}

	if(mode == "kusner") {
		runKusnerExperiment(trainDatasetFilePath, testDatasetFilePath, embeddingsFilePath, k, functionName, r, verbose);
	} else {
		runTripletsExperiment(tripletsFilePath, documentsFilePath, embeddingsFilePath, numClusters, maxIterations, functionName, r, verbose);
	}
	return 0;
}
