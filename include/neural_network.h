#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <Eigen/Dense>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int vocabSize, int embeddingDim, int hiddenDim, int contextLength);
    ~NeuralNetwork();

    Eigen::VectorXd forward(const vector<int>& inputTokens);
    void backward(const Eigen::VectorXd& prediction, const vector<int>& target);
    void updateWeights(double learningRate);

    void saveModel(const string& filename);
    void loadModel(const string& filename);

private:
    int vocabSize;
    int embeddingDim;
    int hiddenDim;
    int contextLength;

    Eigen::MatrixXd embeddingMatrix;
    Eigen::MatrixXd hiddenWeights;
    Eigen::VectorXd hiddenBias;
    Eigen::MatrixXd outputWeights;
    Eigen::VectorXd outputBias;

    Eigen::MatrixXd embeddingGradients;
    Eigen::MatrixXd hiddenWeightsGradients;
    Eigen::VectorXd hiddenBiasGradients;
    Eigen::MatrixXd outputWeightsGradients;
    Eigen::VectorXd outputBiasGradients;

    Eigen::VectorXd embeddings;
    Eigen::VectorXd hiddenActivations;

    void initializeWeights();
    Eigen::VectorXd softmax(const Eigen::VectorXd& input);
    Eigen::VectorXd relu(const Eigen::VectorXd& input);
    Eigen::VectorXd reluDerivative(const Eigen::VectorXd& input);
};

#endif