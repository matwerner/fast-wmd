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

        float computeDistance(const std::vector<std::pair<unsigned int, float>>& nbow1,
                              const std::vector<std::pair<unsigned int, float>>& nbow2) {
            // Source and Target nodes do not share tokens
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
