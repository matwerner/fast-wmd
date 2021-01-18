#ifndef FAST_WMD_C_RELWMD_H
#define FAST_WMD_C_RELWMD_H

#include <memory>
#include <ortools/graph/ebert_graph.h>
#include <ortools/graph/min_cost_flow.h>
#include "C_Distance.h"
#include "C_RelatedWords.h"
#include "C_Util.h"

namespace fastwmd {

    class C_RelWMD: public C_Distance {

    public:

        C_RelWMD() {}

        C_RelWMD(const std::shared_ptr<C_RelatedWords>& relatedWords): m_relatedWords(relatedWords) {}

        /**
         * Compute the Related Word Mover's Distance (Rel-WMD) between Doc1 and Doc2.
         * This distance is equivalent to the Transportation problem in a sparse bipartite graph.
         * The source nodes are the tokens in Doc1, the target nodes are the tokens in Doc2
         * and the edges are composed by the related token pairs between them. The costs of these edges
         * are given by the word embedding distances between each token pair.
         *
         * @param nbow1 L1-Normalized BOW representation of Doc1 (sorted by token index)
         * @param nbow2 L1-Normalized BOW representation of Doc2 (sorted by token index)
         * @return Rel-WMD distance value
         */
        DistanceValue computeDistance(const Document& nbow1, const Document& nbow2) {
            // For a faster execution, documents do not share tokens.
            // E.g: If nbow1 and nbow2 contain the token 'AI' with nbow1['AI'] = 0.1 and nbow2['AI'] = 0.2.
            //      We take the diff so that 'AI' is removed from nbow1 and nbow2['AI'] = 0.1.
            //      This is the same of transporting weight = 0.1 with cost 0.
            const auto& diffNbow1 = C_Util::getHashedDocument(C_Util::diffNbow(nbow1, nbow2));
            const auto& diffNbow2 = C_Util::getHashedDocument(C_Util::diffNbow(nbow2, nbow1));

            // Convert token to node
            operations_research::NodeIndex nodeIndex = 0;
            std::unordered_map<TokenIndex, operations_research::NodeIndex> tokenToNodeMap;
            tokenToNodeMap.reserve(diffNbow1.size() + diffNbow2.size());
            for(const auto& token: diffNbow1)
                tokenToNodeMap[token.first] = nodeIndex++;
            for(const auto& token: diffNbow2)
                tokenToNodeMap[token.first] = nodeIndex++;

            // Find related words in r|D| instead of |D|^2
            const auto& edges = getEdges(diffNbow1, diffNbow2, tokenToNodeMap);

            std::vector<std::pair<operations_research::NodeIndex, float>> nodes;
            nodes.reserve(diffNbow1.size() + diffNbow2.size());
            // Converting supplies
            for(const auto& token: diffNbow1) {
                nodes.emplace_back(tokenToNodeMap[token.first], token.second);
            }
            // Converting demands
            for(const auto& token: diffNbow2) {
                nodes.emplace_back(tokenToNodeMap[token.first], -token.second);
            }
            // Adding an extra node for solution feasible -> there must always be a path connecting sources and targets
            operations_research::NodeIndex transshipmentNodeIndex = diffNbow1.size() + diffNbow2.size();

            // Minimum cost flow
            operations_research::StarGraph graph(nodes.size() + 1, edges.size());
            operations_research::MinCostFlow minCostFlow(&graph);

            // Setting up nodes
            operations_research::NodeIndex maxNodeIndex = -1;
            operations_research::FlowQuantity nodeSupply, deltaSupply = 0, maxNodeSupply = 0;
            for(const auto& node: nodes) {
                nodeIndex = node.first;
                nodeSupply = std::round(MASS_MULT * node.second);
                minCostFlow.SetNodeSupply(nodeIndex, nodeSupply);

                if(nodeSupply > maxNodeSupply) {
                    maxNodeSupply = nodeSupply;
                    maxNodeIndex = nodeIndex;
                }
                deltaSupply += nodeSupply;
            }
            minCostFlow.SetNodeSupply(maxNodeIndex, minCostFlow.Supply(maxNodeIndex) - deltaSupply);
            minCostFlow.SetNodeSupply(transshipmentNodeIndex, 0);

            // Setting up arcs
            for(const auto& edge: edges) {
                operations_research::NodeIndex source = std::get<0>(edge), target = std::get<1>(edge);
                operations_research::CostValue cost = std::round(COST_MULT * std::get<2>(edge));

                operations_research::ArcIndex arc = graph.AddArc(source, target);
                if(source == transshipmentNodeIndex || target == transshipmentNodeIndex) {
                    minCostFlow.SetArcCapacity(arc, std::max(minCostFlow.Supply(source), -minCostFlow.Supply(target)));
                } else {
                    minCostFlow.SetArcCapacity(arc, std::min(minCostFlow.Supply(source), -minCostFlow.Supply(target)));
                }
                minCostFlow.SetArcUnitCost(arc, cost);
            }

            minCostFlow.Solve();
            if(operations_research::MinCostFlow::OPTIMAL != minCostFlow.status()) {
                throw std::runtime_error("operations_research::MinCostFlow::OPTIMAL != minCostFlow.status()");
            }
            return (DistanceValue) (minCostFlow.GetOptimalCost() / (double) (MASS_MULT * COST_MULT));
        }

    private:

        const static int64 MASS_MULT = 1000 * 1000;  // weights quantization constant
        const static int64 COST_MULT = 1000;         // costs quantization constant

        std::shared_ptr<C_RelatedWords> m_relatedWords;

        /**
         * Get all the edges connecting the source (Doc1) to the target (Doc2) nodes.
         *
         * @param diffNbow1 L1-Normalized BOW representation of Doc1 (using hashed map)
         * @param diffNbow2 L1-Normalized BOW representation of Doc2 (using hashed map)
         * @param tokenToNodeMap Structure mapping token indices to node indices
         * @return List of edges
         */
        std::vector<std::tuple<operations_research::NodeIndex, operations_research::NodeIndex, float>> getEdges(
                const HashedDocument& diffNbow1, const HashedDocument& diffNbow2,
                const std::unordered_map<TokenIndex, operations_research::NodeIndex>& tokenToNodeMap) {
            std::size_t r = m_relatedWords->getR();

            // Find related words (edges) between documents in r|D| instead of |D|^2.
            // Instead of checking all token pairs between them, get the related words for each token in one of the documents
            // and for each related word list and each token in it do a look-up operation in the hash table of the other

            // For a faster execution, we iterate over the smaller doc and then do a look-up
            // in the hash table of the larger one.
            bool isSourcesSmaller = diffNbow1.size() < diffNbow2.size();
            const auto& smaller = isSourcesSmaller? diffNbow1 : diffNbow2;
            const auto& larger = isSourcesSmaller? diffNbow2 : diffNbow1;
            std::vector<std::tuple<operations_research::NodeIndex, operations_research::NodeIndex, float>> edges;
            edges.reserve(std::min(r, (std::size_t) larger.size()) * smaller.size());

            for(const auto& smallerIt: smaller) {
                TokenIndex smallerTokenIndex = smallerIt.first;
                operations_research::NodeIndex smallerNodeIndex = tokenToNodeMap.at(smallerTokenIndex);
                const auto& relatedWords = m_relatedWords->getRelatedWords(smallerTokenIndex);

                // We want to find out the tokens in common between the larger document and related words.
                // Since both related words and larger are hashed, iterate over the smaller one of the two to get the
                // tokens in common. Using this approach, we should have have a faster execution.
                if(relatedWords.size() > larger.size()) {
                    for(const auto& largerIt: larger) {
                        TokenIndex largerTokenIndex = largerIt.first;
                        const auto& relatedIt = relatedWords.find(largerTokenIndex);
                        if(relatedIt == relatedWords.end()) continue;

                        operations_research::NodeIndex largerNodeIndex = tokenToNodeMap.at(largerTokenIndex);
                        if(isSourcesSmaller) {
                            edges.emplace_back(smallerNodeIndex, largerNodeIndex, relatedIt->second);
                        } else {
                            edges.emplace_back(largerNodeIndex, smallerNodeIndex, relatedIt->second);
                        }
                    }
                }
                else {
                    for(const auto& relatedIt: relatedWords) {
                        TokenIndex relatedTokenIndex = relatedIt.first;
                        const auto& largerIt = larger.find(relatedTokenIndex);
                        if(largerIt == larger.end()) continue;

                        operations_research::NodeIndex relatedNodeIndex = tokenToNodeMap.at(relatedTokenIndex);
                        if(isSourcesSmaller) {
                            edges.emplace_back(smallerNodeIndex, relatedNodeIndex, relatedIt.second);
                        } else {
                            edges.emplace_back(relatedNodeIndex, smallerNodeIndex, relatedIt.second);
                        }
                    }
                }
            }

            // Since we are removing some edges of the original WMD formulation, the solution can be unfeasible now.
            // To the new formulation be always feasible we have to add alternative edges connecting the source and
            // target nodes. However, we just want to use these edges in last case, thus they have higher edge costs.
            DistanceValue maximumDistance = m_relatedWords->getMaximumDistance();
            operations_research::NodeIndex transshipmentNodeIndex = diffNbow1.size() + diffNbow2.size();
            for(const auto& token: diffNbow1) {
                operations_research::NodeIndex nodeIndex = tokenToNodeMap.at(token.first);
                edges.emplace_back(nodeIndex, transshipmentNodeIndex, maximumDistance);
            }

            for(const auto& token: diffNbow2) {
                operations_research::NodeIndex nodeIndex = tokenToNodeMap.at(token.first);
                edges.emplace_back(transshipmentNodeIndex, nodeIndex, 0);
            }

            return edges;
        }

    };

}


#endif //FAST_WMD_C_RELWMD_H
