#ifndef FAST_WMD_C_EMBEDDINGS_H
#define FAST_WMD_C_EMBEDDINGS_H

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "C_CONSTANTS.h"

namespace fastwmd {

    class C_Embeddings {

    public:

        C_Embeddings() {}

        C_Embeddings(const std::string& filepath, std::size_t maxNumEmbeddings=std::numeric_limits<TokenIndex>::max()) {
            std::ifstream inFile (filepath);
            if (!inFile.is_open()) {
                throw std::runtime_error("Unable to open file. Path: " + filepath);
            }

            std::size_t numEmbeddings;
            inFile >> numEmbeddings >> m_embeddingSize;
            inFile.ignore(); // ignore '\n'

            m_numEmbeddings = std::min(numEmbeddings, maxNumEmbeddings);

            std::string line;
            m_tokens.resize(m_numEmbeddings);
            m_embeddings = EigenDistanceMatrix(m_embeddingSize, m_numEmbeddings);
            std::size_t i;
            for(i = 0; i < m_numEmbeddings && inFile.peek() != std::ifstream::traits_type::eof(); i++) {
                std::string lineError = "Line: " + std::to_string(i) + ". File: " + filepath + "\n";
                try {
                    std::getline(inFile, line);
                    std::stringstream inputStream(line);
                    std::istream_iterator<std::string> begin(inputStream);
                    std::istream_iterator<std::string> end;
                    std::vector<std::string> inputVector(begin, end);

                    if(inputVector.size() != m_embeddingSize + 1) {
                        throw std::runtime_error("Embedding size is " + std::to_string(inputVector.size() - 1) +
                                                 ", but it should be " + std::to_string(m_embeddingSize) + ".\n" + lineError);
                    }

                    // int tokenId = std::stoi(inputVector[0]);
                    m_tokens[i] = inputVector[0];
                    for(std::size_t j = 0; j < m_embeddingSize; j++) {
                        try {
                            m_embeddings(j, i) = (DistanceValue) std::stod(inputVector[j+1]);
                        } catch (const std::exception& e) {
                            throw std::runtime_error("Embedding value could not be parsed.\n" + lineError);
                        }
                    }
                } catch (const std::exception& e) {
                    throw std::runtime_error("Embedding could not be parsed.\n" + lineError);
                }
            }
            inFile.close();

            if(i < m_numEmbeddings || (m_numEmbeddings == numEmbeddings && inFile.peek() != std::ifstream::traits_type::eof())) {
                throw std::runtime_error("Number of tokens is incorrect. File: " + filepath + "\n");
            }
        }

        C_Embeddings(std::size_t numEmbeddings, std::size_t embeddingSize, std::vector<std::vector<DistanceValue>>& embeddings) {
            m_numEmbeddings = numEmbeddings;
            m_embeddingSize = embeddingSize;
            m_embeddings = EigenDistanceMatrix(m_embeddingSize, m_numEmbeddings);
            for(std::size_t i = 0; i < m_numEmbeddings; i++) {
                m_embeddings.col(i) = Eigen::Map<EigenDistanceVector, Eigen::Unaligned>(embeddings[i].data(), embeddings[i].size());
            }
        }

        std::size_t getNumEmbeddings() {
            return m_numEmbeddings;
        }

        std::size_t getEmbeddingSize() {
            return m_embeddingSize;
        }

        const std::vector<std::string>& getTokens() {
            return m_tokens;
        }

        const EigenDistanceMatrix& getEmbeddings() {
            return m_embeddings;
        }

        EigenDistanceMatrix getEmbeddings(const std::vector<TokenIndex>& indices) {
            EigenDistanceMatrix S(m_embeddingSize, indices.size());
            for (std::size_t i = 0; i < indices.size(); i++) {
                S.col(i) = m_embeddings.col(indices[i]);
            }
            return S;
        }

        EigenDistanceVector computeDistances(TokenIndex index) {
            EigenDistanceVector SS = m_embeddings.colwise().squaredNorm();
            EigenDistanceVector TT = m_embeddings.col(index).squaredNorm() * EigenDistanceVector::Ones(m_numEmbeddings);
            EigenDistanceVector ST = 2 * m_embeddings.transpose() * m_embeddings.col(index);
            return (SS - ST + TT).cwiseSqrt();
        }

        EigenDistanceMatrix computeDistances(const std::vector<TokenIndex>& indices1, const std::vector<TokenIndex>& indices2) {
            EigenDistanceMatrix S = getEmbeddings(indices1), T = getEmbeddings(indices2);

            EigenDistanceMatrix SS = EigenDistanceMatrix::Zero(indices1.size(), indices2.size()).colwise()
                                 + S.colwise().squaredNorm().transpose();
            EigenDistanceMatrix TT = EigenDistanceMatrix::Zero(indices1.size(), indices2.size()).array().rowwise()
                                 + T.colwise().squaredNorm().array();
            EigenDistanceMatrix ST = 2 * (S.transpose() * T);
            return (SS - ST + TT).cwiseSqrt();
        }

    private:

        std::size_t m_numEmbeddings, m_embeddingSize;
        std::vector<std::string> m_tokens;
        EigenDistanceMatrix m_embeddings;

    };

}

#endif //FAST_WMD_C_EMBEDDINGS_H
