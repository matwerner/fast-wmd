#include <ctime>
#include <iostream>
#include <ortools/graph/ebert_graph.h>
#include <ortools/graph/min_cost_flow.h>
#include <ortools/graph/max_flow.h>

#include "Distances.hpp"
#include "Tools.hpp"

/**
 * Linear Complexity Relaxed Word Mover's Distance + Edges reduction implementation.
 */
float Distances::computeLinearRelatedRWMD(const Eigen::SparseVector<int> &doc1, const std::unordered_map<int, int> &nearDistancesToDoc1,
                                          const Eigen::SparseVector<int> &doc2, const std::unordered_map<int, int> &nearDistancesToDoc2,
                                          int maximumDistance) {
    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;
    int sourcesSize = 0, targetsSize = 0;

    // Split words in Doc1 and Doc2 - after eliminating common tokens
    float l1 = 0.0f, l2 = 0.0f;
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            const auto it2 = nearDistancesToDoc2.find(it.row());
            if(it2 == nearDistancesToDoc2.end()) {
                l1 += it.value() * maximumDistance;
            } else {
                l1 += it.value() * it2->second;
            }
            sourcesSize++;
        }
        else {
            const auto it2 = nearDistancesToDoc1.find(it.row());
            if(it2 == nearDistancesToDoc1.end()) {
                l2 -= it.value() * maximumDistance;
            } else {
                l2 -= it.value() * it2->second;
            }
            targetsSize++;
        }
    }

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    return std::max(l1, l2);
}

/**
 * Relaxed Word Mover's Distance + Edges reduction implementation.
 */
float Distances::computeRelRWMD(const Eigen::SparseVector<int> &doc1, const std::unordered_map<int, int> &doc1Hash,
                                const Eigen::SparseVector<int> &doc2, const std::unordered_map<int, int> &doc2Hash,
                                const std::pair<std::vector<std::unordered_map<int, int>>, int>& relatedWordsCache) {
    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;
    int sourcesSize = 0, targetsSize = 0;

    const auto& relatedWordsDict = relatedWordsCache.first;
    int maximumDistance = relatedWordsCache.second;

    float l1 = 0.0f, l2 = 0.0f;
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        int minDistance = maximumDistance;
        const auto& relatedWords = relatedWordsDict[it.row()];
        if(it.value() > 0) {
            if(doc2.nonZeros() < relatedWords.size()) {
                for (Eigen::SparseVector<int>::InnerIterator doc2It(doc2); doc2It; ++doc2It) {
                    const auto& relatedIt = relatedWords.find(doc2It.row());
                    if(relatedIt == relatedWords.end()) continue;
                    minDistance = std::min(minDistance, relatedIt->second);
                }
            } else {
                for(const auto& relatedIt: relatedWords) {
                    const auto& doc2It = doc2Hash.find(relatedIt.first);
                    if(doc2It == doc2Hash.end()) continue;
                    minDistance = std::min(minDistance, relatedIt.second);
                }
            }
            l1 += it.value() * minDistance;
            sourcesSize++;
        } else {
            if(doc1.nonZeros() < relatedWords.size()) {
                for (Eigen::SparseVector<int>::InnerIterator doc1It(doc1); doc1It; ++doc1It) {
                    const auto& relatedIt = relatedWords.find(doc1It.row());
                    if(relatedIt == relatedWords.end()) continue;
                    minDistance = std::min(minDistance, relatedIt->second);
                }
            } else {
                for(const auto& relatedIt: relatedWords) {
                    const auto& doc1It = doc1Hash.find(relatedIt.first);
                    if(doc1It == doc1Hash.end()) continue;
                    minDistance = std::min(minDistance, relatedIt.second);
                }
            }
            l2 -= it.value() * minDistance;
            targetsSize++;
        }
    }

    /*
    float l1 = 0.0f;
    for(const auto& doc1It: doc1Hash) {
        int weight = doc1It.second, minDistance = maximumDistance;
        const auto& it = doc2Hash.find(doc1It.first);
        weight -= it == doc2Hash.end()? 0 :  it->second;
        if(weight <= 0) continue;

        const auto& relatedWords = relatedWordsDict[doc1It.first];
        if(doc2.size() < relatedWords.size()) {
            for (const auto& doc2It: doc2Hash) {
                const auto& relatedIt = relatedWords.find(doc2It.first);
                if(relatedIt == relatedWords.end()) continue;
                minDistance = std::min(minDistance, relatedIt->second);
            }
        } else {
            for(const auto& relatedIt: relatedWords) {
                const auto& doc2It = doc2Hash.find(relatedIt.first);
                if(doc2It == doc2Hash.end()) continue;
                minDistance = std::min(minDistance, relatedIt.second);
            }
        }
        l1 += weight * minDistance;
        sourcesSize++;
    }

    float l2 = 0.0f;
    for(const auto& doc2It: doc2Hash) {
        int weight = doc2It.second, minDistance = maximumDistance;
        const auto& it = doc1Hash.find(doc2It.first);
        weight -= it == doc1Hash.end()? 0 :  it->second;
        if(weight <= 0) continue;

        const auto& relatedWords = relatedWordsDict[doc2It.first];
        if(doc1.size() < relatedWords.size()) {
            for (const auto& doc1It: doc1Hash) {
                const auto& relatedIt = relatedWords.find(doc1It.first);
                if(relatedIt == relatedWords.end()) continue;
                minDistance = std::min(minDistance, relatedIt->second);
            }
        } else {
            for(const auto& relatedIt: relatedWords) {
                const auto& doc1It = doc1Hash.find(relatedIt.first);
                if(doc1It == doc1Hash.end()) continue;
                minDistance = std::min(minDistance, relatedIt.second);
            }
        }
        l2 += weight * minDistance;
        targetsSize++;
    }
    */

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    return std::max(l1, l2);
}

/**
 * Word Mover's Distance + Edges reduction implementation.
 * Following formulation proposed by Pele et al. ("Fast and robust Earth Mover's Distances").
 */
float Distances::computeRelWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2, int r,
                               const std::pair<std::vector<std::unordered_map<int, int>>, int>& relatedWordsCache) {
    int doc1Size = doc1.nonZeros(), doc2Size = doc2.nonZeros();

    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;

    int maximumFlow = 0, maximumDistance = relatedWordsCache.second;
    int nonZerosSize = diff.nonZeros(), nodeIndex = 0, sourcesSize = 0, targetsSize = 0;

    // Split words between sources and targets
    std::unordered_map<int, std::pair<int,int>> sourceTokens; sourceTokens.reserve(doc1Size);
    std::unordered_map<int, std::pair<int,int>> targetTokens; targetTokens.reserve(doc2Size);
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            maximumFlow += it.value();
            sourceTokens[it.row()] = std::make_pair(nodeIndex, it.value());
            sourcesSize++;
        }
        else {
            targetTokens[it.row()] = std::make_pair(nodeIndex, -it.value());
            targetsSize++;
        }
        nodeIndex++;
    }

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    // Find related words in k|d| instead of |d|^2
    const auto& relatedWordsDict = relatedWordsCache.first;
    bool isSourcesSmaller = sourceTokens.size() < targetTokens.size();
    const auto& smaller = isSourcesSmaller? sourceTokens : targetTokens;
    const auto& larger = isSourcesSmaller? targetTokens : sourceTokens;
    std::vector<std::pair<int, int>> relatedDistancePairs; relatedDistancePairs.reserve(std::min(r, (int) larger.size())
                                                                                        * smaller.size());
    for(const auto& smallerIt: smaller) {
        int smallerTokenIndex = smallerIt.first;
        const auto& relatedWords = relatedWordsDict[smallerTokenIndex];

        // r is bigger than document size
        if(relatedWords.size() > larger.size()) {
            for(const auto& largerIt: larger) {
                int largerTokenIndex = largerIt.first;
                const auto& relatedIt = relatedWords.find(largerTokenIndex);
                if(relatedIt == relatedWords.end()) continue;
                relatedDistancePairs.emplace_back(std::make_pair(smallerTokenIndex, largerTokenIndex));
            }
        }
        else {
            for(const auto& relatedIt: relatedWords) {
                int relatedTokenIndex = relatedIt.first;
                const auto& largerIt = larger.find(relatedTokenIndex);
                if(largerIt == larger.end()) continue;
                relatedDistancePairs.emplace_back(std::make_pair(smallerTokenIndex, relatedTokenIndex));
            }
        }
    }

    // No related words, No flow
    if(relatedDistancePairs.empty()) {
        return maximumDistance * (float) maximumFlow;
    }

    // Max Flow
    int transshipmentNode = sourcesSize + targetsSize;
    operations_research::StarGraph graph(sourcesSize + targetsSize + 1,
                                         sourcesSize + targetsSize + relatedDistancePairs.size());
    operations_research::MinCostFlow minCostFlow(&graph);

    // Setting up source supply
    minCostFlow.SetNodeSupply(transshipmentNode, 0);
    for (const auto& it: sourceTokens) {
        int sourceIndex = it.second.first, sourceWeight = it.second.second;
        minCostFlow.SetNodeSupply(sourceIndex, sourceWeight);
        operations_research::ArcIndex arc = graph.AddArc(sourceIndex, transshipmentNode);
        minCostFlow.SetArcCapacity(arc, sourceWeight);
        minCostFlow.SetArcUnitCost(arc, maximumDistance);
    }

    // Setting up target demand
    for(const auto& it: targetTokens) {
        int targetIndex = it.second.first, targetWeight = it.second.second;
        minCostFlow.SetNodeSupply(targetIndex, -targetWeight);
        operations_research::ArcIndex arc = graph.AddArc(transshipmentNode, targetIndex);
        minCostFlow.SetArcCapacity(arc, targetWeight);
        minCostFlow.SetArcUnitCost(arc, 0);
    }

    // Setting up arcs
    for(const auto& pair: relatedDistancePairs) {
        int sourceTokenIndex = isSourcesSmaller? pair.first : pair.second;
        int targetTokenIndex = isSourcesSmaller? pair.second : pair.first;

        const auto& sourceToken = sourceTokens[sourceTokenIndex];
        const auto& targetToken = targetTokens[targetTokenIndex];
        operations_research::ArcIndex arc = graph.AddArc(sourceToken.first, targetToken.first);
        minCostFlow.SetArcCapacity(arc, std::min(sourceToken.second, targetToken.second));
        minCostFlow.SetArcUnitCost(arc, relatedWordsDict[sourceTokenIndex].at(targetTokenIndex));
    }

    minCostFlow.Solve();
    if(operations_research::MinCostFlow::OPTIMAL != minCostFlow.status()) {
        throw std::runtime_error("operations_research::MinCostFlow::OPTIMAL == minCostFlow.status()");
    }

    return minCostFlow.GetOptimalCost();
}

/**
 * Word Centroid Distance implementation.
 */
Eigen::VectorXi Distances::computeWCD(const Eigen::SparseVector<int>& doc, const Eigen::MatrixXf& embeddings) {
    return (embeddings * doc.cast<float>()).cast<int>();
}

/**
 * Word Centroid Distance implementation.
 */
Eigen::MatrixXi Distances::computeWCD(const Eigen::SparseMatrix<int>& docs, const Eigen::MatrixXf& embeddings) {
    return (embeddings * docs.cast<float>()).cast<int>();
}

/**
 * Linear Complexity Relaxed Word Mover's Distance implementation.
 */
float Distances::computeLinearRWMD(const Eigen::SparseVector<int> &doc1, const Eigen::VectorXi &nearDistancesToDoc1,
                                   const Eigen::SparseVector<int> &doc2, const Eigen::VectorXi &nearDistancesToDoc2) {
    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;
    int sourcesSize = 0, targetsSize = 0;

    // Split words in Doc1 and Doc2 - after eliminating common tokens
    float l1 = 0.0f, l2 = 0.0f;
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            l1 += it.value() * nearDistancesToDoc2[it.row()];
            sourcesSize++;
        }
        else {
            l2 -= it.value() * nearDistancesToDoc1[it.row()];
            targetsSize++;
        }
    }

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    return std::max(l1, l2);
}

/**
 * Relaxed Word Mover's Distance implementation.
 */
float Distances::computeRWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2,
                  const Eigen::MatrixXf& embeddings) {
    int doc1Size = doc1.nonZeros(), doc2Size = doc2.nonZeros();

    Eigen::MatrixXf S(embeddings.rows(), doc1Size), T(embeddings.rows(), doc2Size);
    Eigen::VectorXf sourceWeights(doc1Size), targetWeights(doc2Size);

    int i = 0;
    for (Eigen::SparseVector<int>::InnerIterator it(doc1); it; ++it) {
        sourceWeights[i] = it.value();
        S.col(i) = embeddings.col(it.row());
        i++;
    }
    int j = 0;
    for (Eigen::SparseVector<int>::InnerIterator it(doc2); it; ++it) {
        targetWeights[j] = it.value();
        T.col(j) = embeddings.col(it.row());
        j++;
    }

    Eigen::MatrixXi D = Tools::computeEuclideanDistanceBetweenDocsEmbeddings(S,T);

    /*
    Eigen::VectorXf minSourceDistances = Eigen::VectorXf::Constant(doc1Size, std::numeric_limits<float>::max());
    Eigen::VectorXf minTargetDistances = Eigen::VectorXf::Constant(doc2Size, std::numeric_limits<float>::max());

    i = 0;
    for (Eigen::SparseVector<int>::InnerIterator it(doc1); it; ++it) {
        j = 0;
        for (Eigen::SparseVector<int>::InnerIterator it2(doc2); it2; ++it2) {
            if(it.row() == it2.row()) continue;
            float distance = D(i,j);
            minSourceDistances[i] = std::min(minSourceDistances[i], distance);
            minTargetDistances[j] = std::min(minTargetDistances[j], distance);
            j++;
        }
        i++;
    }

    float l1 = minTargetDistances.transpose() * targetWeights;
    float l2 = sourceWeights.transpose() * minSourceDistances;
    */

    Eigen::SparseVector<int> diff = doc1 - doc2;

    int sourcesSize = 0, targetsSize = 0, doc1Idx = 0, doc2Idx = 0;
    float l1 = 0.0f, l2 = 0.0f;
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            j = 0;
            float minDistance = std::numeric_limits<float>::max();
            for (Eigen::SparseVector<int>::InnerIterator it2(doc2); it2; ++it2) {
                if(it.row() != it2.row()) {
                    float distance =  D(doc1Idx, j);
                    minDistance = std::min(minDistance, distance);
                } else {
                    doc2Idx++;
                }
                j++;
            }
            l1 += it.value() * minDistance;
            sourcesSize++; doc1Idx++;
        }
        else {
            i = 0;
            float minDistance = std::numeric_limits<float>::max();
            for (Eigen::SparseVector<int>::InnerIterator it2(doc1); it2; ++it2) {
                if(it.row() != it2.row()) {
                    float distance =  D(i, doc2Idx);
                    minDistance = std::min(minDistance, distance);
                } else {
                    doc1Idx++;
                }
                i++;
            }
            l2 -= it.value() * minDistance;
            targetsSize++; doc2Idx++;
        }
    }

    return std::max(l1, l2);

    /*
    // CHECK FOR BUG - IF NOTHING IS FOUND, REMOVE THIS SECTION

    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;

    int sourcesSize = 0, sourcesExtraSize = 0, targetsSize = 0, targetsExtraSize = 0;

    Eigen::VectorXf sourceWeights(doc1Size), targetWeights(doc2Size);
    Eigen::MatrixXf S(embeddings.rows(), doc1Size), T(embeddings.rows(), doc2Size);
    Eigen::MatrixXf SExtra(embeddings.rows(), doc1Size), TExtra(embeddings.rows(), doc2Size);
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            sourceWeights[sourcesSize] = it.value();
            S.col(sourcesSize) = embeddings.col(it.row());
            sourcesSize++;
            if(doc2.coeff(it.row()) > 0) {
                TExtra.col(sourcesExtraSize) = embeddings.col(it.row());
                targetsExtraSize++;
            }
        }
        else {
            targetWeights[targetsSize] = -it.value();
            T.col(targetsSize) = embeddings.col(it.row());
            targetsSize++;
            if(doc1.coeff(it.row()) > 0) {
                SExtra.col(sourcesExtraSize) = embeddings.col(it.row());
                sourcesExtraSize++;
            }
        }
    }

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    S.conservativeResize(Eigen::NoChange, sourcesSize);
    T.conservativeResize(Eigen::NoChange, targetsSize);
    SExtra.conservativeResize(Eigen::NoChange, sourcesExtraSize);
    TExtra.conservativeResize(Eigen::NoChange, targetsExtraSize);
    sourceWeights.conservativeResize(sourcesSize);
    targetWeights.conservativeResize(targetsSize);

    Eigen::MatrixXi DInCommon = Tools::computeEuclideanDistanceBetweenDocsEmbeddings(S,T);
    Eigen::MatrixXi DFromS(sourcesSize, targetsSize + targetsExtraSize), DFromT(sourcesSize + sourcesExtraSize, targetsSize);
    DFromS << DInCommon, Tools::computeEuclideanDistanceBetweenDocsEmbeddings(S,TExtra);
    DFromT << DInCommon, Tools::computeEuclideanDistanceBetweenDocsEmbeddings(SExtra,T);

    float l1 = DFromT.colwise().minCoeff().cast<float>() * targetWeights;
    float l2 = sourceWeights.transpose() * DFromS.rowwise().minCoeff().cast<float>();

    return std::max(l1, l2);
    */
}

/**
 * Word Mover's Distance implementation.
 */
float Distances::computeWMD(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2,
                 const Eigen::MatrixXf& embeddings) {
    int doc1Size = doc1.nonZeros(), doc2Size = doc2.nonZeros();

    // Unique tokens
    Eigen::SparseVector<int> diff = doc1 - doc2;

    int nonZerosSize = diff.nonZeros(), nodeIndex = 0, sourcesSize = 0, targetsSize = 0;
    operations_research::StarGraph graph(doc1Size + doc2Size, doc1Size * doc2Size);
    operations_research::MinCostFlow minCostFlow(&graph);

    Eigen::VectorXi sourceIndices(doc1Size), targetIndices(doc2Size);
    Eigen::MatrixXf S(embeddings.rows(), doc1Size), T(embeddings.rows(), doc2Size);
    for (Eigen::SparseVector<int>::InnerIterator it(diff); it; ++it) {
        if(it.value() > 0) {
            sourceIndices[sourcesSize] = nodeIndex;
            minCostFlow.SetNodeSupply(nodeIndex, it.value());
            S.col(sourcesSize) = embeddings.col(it.row());
            sourcesSize++;
        }
        else {
            targetIndices[targetsSize] = nodeIndex;
            minCostFlow.SetNodeSupply(nodeIndex, it.value());
            T.col(targetsSize) = embeddings.col(it.row());
            targetsSize++;
        }
        nodeIndex++;
    }

    if(sourcesSize == 0 || targetsSize == 0) {
        return 0.0f;
    }

    S.conservativeResize(Eigen::NoChange, sourcesSize);
    T.conservativeResize(Eigen::NoChange, targetsSize);
    sourceIndices.conservativeResize(sourcesSize);
    targetIndices.conservativeResize(targetsSize);

    // Setting up arcs
    Eigen::MatrixXi D = Tools::computeEuclideanDistanceBetweenDocsEmbeddings(S,T);
    for(int i = 0; i < sourcesSize; i++) {
        for(int j = 0; j < targetsSize; j++) {
            int source = sourceIndices[i], target = targetIndices[j];
            operations_research::ArcIndex arc = graph.AddArc(source, target);
            minCostFlow.SetArcUnitCost(arc, D(i,j));
            minCostFlow.SetArcCapacity(arc, std::min(minCostFlow.Supply(source), -minCostFlow.Supply(target)));
        }
    }

    minCostFlow.Solve();
    if(operations_research::MinCostFlow::OPTIMAL != minCostFlow.status()) {
        throw std::runtime_error("operations_research::MinCostFlow::OPTIMAL == minCostFlow.status()");
    }

    return minCostFlow.GetOptimalCost();
}

/**
 * Cosine distance implementation.
 */
float Distances::computeCosineDistance(const Eigen::SparseVector<int> &doc1, const Eigen::SparseVector<int> &doc2) {
    return 1.0f - doc1.cast<float>().dot(doc2.cast<float>())/(doc1.cast<float>().norm() * doc2.cast<float>().norm());
}

/**
 * Cosine distance implementation.
 */
float Distances::computeEuclideanDistance(const Eigen::VectorXi &doc1, const Eigen::VectorXi &doc2) {
    return (doc1.cast<float>() - doc2.cast<float>()).norm();
}