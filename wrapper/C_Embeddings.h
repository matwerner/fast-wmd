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

        /**
         * Load Word embeddings from file in "gensim" format.
         * Format:
         * <Num tokens> <Word embedding dimensionality>
         * <Token1> <emb11> <emb12> ... <emb1N>
         * ...
         * <TokenM> <embM1> <embM2> ... <embMN>
         * @param filepath Word embeddings filepath
         * @param maxNumEmbeddings Maximum number of word embeddings to load from file
         */
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
            m_embeddings = EigenEmbeddingMatrix(m_embeddingSize, m_numEmbeddings);
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
                            m_embeddings(j, i) = (EmbeddingWeight) std::stod(inputVector[j+1]);
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

        /**
         * Store word embeddings data
         * 
         * @param numEmbeddings Number of word embeddings
         * @param embeddingSize Word embedding dimensionality
         * @param embeddings Raw word embeddings data
         */
        C_Embeddings(std::size_t numEmbeddings, std::size_t embeddingSize, std::vector<std::vector<EmbeddingWeight>>& embeddings) {
            m_numEmbeddings = numEmbeddings;
            m_embeddingSize = embeddingSize;
            m_embeddings = EigenEmbeddingMatrix(m_embeddingSize, m_numEmbeddings);
            for(std::size_t i = 0; i < m_numEmbeddings; i++) {
                m_embeddings.col(i) = Eigen::Map<EigenEmbeddingVector, Eigen::Unaligned>(embeddings[i].data(), embeddings[i].size());
            }
        }

        /**
         * Get number of word embeddings
         * 
         * @return Number of word embeddings
         */
        std::size_t getNumEmbeddings() {
            return m_numEmbeddings;
        }

        /**
         * Get word embedding dimensionality
         * 
         * @return Word embedding dimensionality
         */
        std::size_t getEmbeddingSize() {
            return m_embeddingSize;
        }

        /**
         * Get list of tokens related to the word embedding stored
         * 
         * @return List of tokens
         */
        const std::vector<std::string>& getTokens() {
            return m_tokens;
        }

        /**
         * Get all word embeddings
         * 
         * @return Word embedding matrix in column-wise format
         */
        const EigenEmbeddingMatrix& getEmbeddings() {
            return m_embeddings;
        }

        /**
         * Get word embeddings from the given token indices
         * 
         * @param indices List of token indices
         * @return Word embedding matrix in column-wise format
         */
        EigenEmbeddingMatrix getEmbeddings(const std::vector<TokenIndex>& indices) {
            EigenEmbeddingMatrix S(m_embeddingSize, indices.size());
            for (std::size_t i = 0; i < indices.size(); i++) {
                S.col(i) = m_embeddings.col(indices[i]);
            }
            return S;
        }

        /**
         * Compute the distance from one word embeddings to all other word embeddings
         * 
         * @param index Token index
         * @return Word embedding distance vector
         */
        EigenDistanceVector computeDistances(TokenIndex index) {
            EigenDistanceVector SS = m_embeddings.colwise().squaredNorm();
            EigenDistanceVector TT = m_embeddings.col(index).squaredNorm() * EigenDistanceVector::Ones(m_numEmbeddings);
            EigenDistanceVector ST = 2 * m_embeddings.transpose() * m_embeddings.col(index);
            return (SS - ST + TT).cwiseSqrt();
        }

        /**
         * Compute the distances from word embeddings in the first list to the ones in the second list
         * 
         * @param indices1 First list of token indices
         * @param indices2 Second list of token indices
         * @return Word embedding distance matrix
         */
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
        EigenEmbeddingMatrix m_embeddings;

    };

}

#endif //FAST_WMD_C_EMBEDDINGS_H
