#ifndef FAST_WMD_C_RELWMD_H
#define FAST_WMD_C_RELWMD_H

#include <memory>
#include <ortools/graph/ebert_graph.h>
#include <ortools/graph/min_cost_flow.h>
#include "C_Distance.h"
#include "C_RelatedWords.h"
#include "C_Util.h"

namespace fastwmd {

    class C_RelWMD: public C_Distance<std::unordered_map<unsigned int, float>> {

    public:

        C_RelWMD() {}

        C_RelWMD(const std::shared_ptr<C_RelatedWords>& relatedWords): m_relatedWords(relatedWords) {}

        float computeDistance(const std::unordered_map<unsigned int, float>& nbow1,
                              const std::unordered_map<unsigned int, float>& nbow2) {
            // Source and Target nodes do not share tokens
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);

            // Convert token to node
            operations_research::NodeIndex nodeIndex = 0;
            std::unordered_map<unsigned int, operations_research::NodeIndex> tokenToNodeMap;
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
            return minCostFlow.GetOptimalCost() / (float) (MASS_MULT * COST_MULT);
        }

    private:

        const static int MASS_MULT = 1000 * 1000;   // weights quantization constant
        const static int COST_MULT = 1000;		 	// costs quantization constant

        std::shared_ptr<C_RelatedWords> m_relatedWords;

        std::vector<std::tuple<operations_research::NodeIndex, operations_research::NodeIndex, float>> getEdges(
                const std::unordered_map<unsigned int, float>& diffNbow1, const std::unordered_map<unsigned int, float>& diffNbow2,
                const std::unordered_map<unsigned int, operations_research::NodeIndex>& tokenToNodeMap) {
            unsigned int r = m_relatedWords->getR();

            // Find related words in r|D| instead of |D|^2
            bool isSourcesSmaller = diffNbow1.size() < diffNbow2.size();
            const auto& smaller = isSourcesSmaller? diffNbow1 : diffNbow2;
            const auto& larger = isSourcesSmaller? diffNbow2 : diffNbow1;
            std::vector<std::tuple<operations_research::NodeIndex, operations_research::NodeIndex, float>> edges;
            edges.reserve(std::min(r, (unsigned int) larger.size()) * smaller.size());

            for(const auto& smallerIt: smaller) {
                unsigned int smallerTokenIndex = smallerIt.first;
                operations_research::NodeIndex smallerNodeIndex = tokenToNodeMap.at(smallerTokenIndex);
                const auto& relatedWords = m_relatedWords->getRelatedWords(smallerTokenIndex);

                // r is bigger than document size
                if(relatedWords.size() > larger.size()) {
                    for(const auto& largerIt: larger) {
                        unsigned int largerTokenIndex = largerIt.first;
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
                        unsigned int relatedTokenIndex = relatedIt.first;
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

            float maximumDistance = m_relatedWords->getMaximumDistance();
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
