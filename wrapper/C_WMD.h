#ifndef FAST_WMD_C_WMD_H
#define FAST_WMD_C_WMD_H

#include <ortools/graph/ebert_graph.h>
#include <ortools/graph/min_cost_flow.h>
#include <memory>
#include "C_Embeddings.h"
#include "C_Distance.h"
#include "C_Util.h"

namespace fastwmd {

    class C_WMD: public C_Distance<std::vector<std::pair<unsigned int, float>>> {

    public:

        C_WMD() {}

        C_WMD(const std::shared_ptr<C_Embeddings>& embeddings): m_embeddings(embeddings) {}

        /**
         * Compute the Word Mover's Distance (WMD) between document 1 and document 2.
         * This distance is equivalent to the Transportation problem in a complete bipartite graph.
         * The source nodes are the tokens in document 1, the target nodes are the tokens in document 2
         * and the edge costs are given by the word embedding distances between each token pair.
         *
         * @param nbow1 List of tokens in Document 1 and their respective weights
         * @param nbow2 List of tokens in Document 2 and their respective weights
         * @return WMD distance
         */
        float computeDistance(const std::vector<std::pair<unsigned int, float>>& nbow1,
                              const std::vector<std::pair<unsigned int, float>>& nbow2) {
            // For a faster execution, documents do not share tokens.
            // E.g: If nbow1 and nbow2 contain the token 'AI' with nbow1['AI'] = 0.1 and nbow2['AI'] = 0.2.
            //      We take the diff so that 'AI' is removed from nbow1 and nbow2['AI'] = 0.1.
            //      This is the same of transporting weight = 0.1 with cost 0.
            const auto& diffNbow1 = C_Util::diffNbow(nbow1, nbow2);
            const auto& diffNbow2 = C_Util::diffNbow(nbow2, nbow1);

            // Minimum cost flow
            operations_research::StarGraph graph(diffNbow1.size() + diffNbow2.size(), diffNbow1.size() * diffNbow2.size());
            operations_research::MinCostFlow minCostFlow(&graph);

            // Setting up supplies
            operations_research::NodeIndex nodeIndex, maxNodeIndex = -1;
            operations_research::FlowQuantity deltaSupply = 0, nodeSupply, maxNodeSupply = 0;
            for(unsigned int i = 0; i < diffNbow1.size(); i++) {
                nodeIndex = i;
                nodeSupply = std::round(MASS_MULT * diffNbow1[i].second);
                if(nodeSupply > maxNodeSupply) {
                    maxNodeSupply = nodeSupply;
                    maxNodeIndex = nodeIndex;
                }
                minCostFlow.SetNodeSupply(nodeIndex, nodeSupply);
                deltaSupply += nodeSupply;
            }

            // Setting up demands
            for(unsigned int j = 0; j < diffNbow2.size(); j++) {
                nodeIndex = diffNbow1.size() + j;
                nodeSupply = std::round(MASS_MULT * diffNbow2[j].second);
                minCostFlow.SetNodeSupply(nodeIndex, -nodeSupply);
                deltaSupply -= nodeSupply;
            }
            minCostFlow.SetNodeSupply(maxNodeIndex, minCostFlow.Supply(maxNodeIndex) - deltaSupply);

            // Setting up arcs
            std::vector<unsigned int> tokens1 = C_Util::getNbowIndices(diffNbow1), tokens2 = C_Util::getNbowIndices(diffNbow2);
            Eigen::MatrixXf D = m_embeddings->computeDistances(tokens1, tokens2);
            for(unsigned int i = 0; i < diffNbow1.size(); i++) {
                operations_research::NodeIndex source = i;
                for(unsigned int j = 0; j < diffNbow2.size(); j++) {
                    operations_research::NodeIndex target = diffNbow1.size() + j;
                    operations_research::CostValue cost = std::round(COST_MULT * D.coeffRef(i,j));

                    operations_research::ArcIndex arc = graph.AddArc(source, target);
                    minCostFlow.SetArcUnitCost(arc, cost);
                    minCostFlow.SetArcCapacity(arc, std::min(minCostFlow.Supply(source), -minCostFlow.Supply(target)));
                }
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

        std::shared_ptr<C_Embeddings> m_embeddings;

    };

}


#endif //FAST_WMD_C_WMD_H
