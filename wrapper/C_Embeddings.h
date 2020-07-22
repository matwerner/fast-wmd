#ifndef FAST_WMD_C_EMBEDDINGS_H
#define FAST_WMD_C_EMBEDDINGS_H

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastwmd {

    class C_Embeddings {

    public:

        C_Embeddings() {}

        C_Embeddings(const std::string& filepath, unsigned int maxNumEmbeddings=std::numeric_limits<unsigned int>::max()) {
            std::ifstream inFile (filepath);
            if (!inFile.is_open()) {
                throw std::runtime_error("Unable to open file. Path: " + filepath);
            }

            unsigned int numEmbeddings;
            inFile >> numEmbeddings >> m_embeddingSize;
            inFile.ignore(); // ignore '\n'

            m_numEmbeddings = std::min(numEmbeddings, maxNumEmbeddings);

            std::string line;
            m_tokens.resize(m_numEmbeddings);
            m_embeddings = Eigen::MatrixXf(m_embeddingSize, m_numEmbeddings);
            unsigned int i;
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
                    for(unsigned int j = 0; j < m_embeddingSize; j++) {
                        try {
                            m_embeddings(j, i) = std::stof(inputVector[j+1]);
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

        C_Embeddings(unsigned int numEmbeddings, unsigned int embeddingSize, std::vector<std::vector<float>>& embeddings) {
            m_numEmbeddings = numEmbeddings;
            m_embeddingSize = embeddingSize;
            m_embeddings = Eigen::MatrixXf(m_embeddingSize, m_numEmbeddings);
            for(unsigned int i = 0; i < m_numEmbeddings; i++) {
                m_embeddings.col(i) = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(embeddings[i].data(), embeddings[i].size());
            }
        }

        unsigned int getNumEmbeddings() {
            return m_numEmbeddings;
        }

        unsigned int getEmbeddingSize() {
            return m_embeddingSize;
        }

        const std::vector<std::string>& getTokens() {
            return m_tokens;
        }

        const Eigen::MatrixXf& getEmbeddings() {
            return m_embeddings;
        }

        Eigen::MatrixXf getEmbeddings(const std::vector<unsigned int>& indices) {
            Eigen::MatrixXf S(m_embeddingSize, indices.size());
            for (unsigned int i = 0; i < indices.size(); i++) {
                S.col(i) = m_embeddings.col(indices[i]);
            }
            return S;
        }

        Eigen::VectorXf computeDistances(unsigned int index) {
            Eigen::VectorXf SS = m_embeddings.colwise().squaredNorm();
            Eigen::VectorXf TT = m_embeddings.col(index).squaredNorm() * Eigen::VectorXf::Ones(m_numEmbeddings);
            Eigen::VectorXf ST = 2 * m_embeddings.transpose() * m_embeddings.col(index);
            return (SS - ST + TT).cwiseSqrt();
        }

        Eigen::MatrixXf computeDistances(const std::vector<unsigned int>& indices1,
                                         const std::vector<unsigned int>& indices2) {
            Eigen::MatrixXf S = getEmbeddings(indices1), T = getEmbeddings(indices2);

            Eigen::MatrixXf SS = Eigen::MatrixXf::Zero(indices1.size(), indices2.size()).colwise()
                                 + S.colwise().squaredNorm().transpose();
            Eigen::MatrixXf TT = Eigen::MatrixXf::Zero(indices1.size(), indices2.size()).array().rowwise()
                                 + T.colwise().squaredNorm().array();
            Eigen::MatrixXf ST = 2 * (S.transpose() * T);
            return (SS - ST + TT).cwiseSqrt();
        }

    private:

        unsigned int m_numEmbeddings, m_embeddingSize;
        std::vector<std::string> m_tokens;
        Eigen::MatrixXf m_embeddings;

    };

}

#endif //FAST_WMD_C_EMBEDDINGS_H
