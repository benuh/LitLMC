#include "neural_network.h"
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

NeuralNetwork::NeuralNetwork(int vocabSize, int embeddingDim, int hiddenDim, int contextLength)
    : vocabSize(vocabSize), embeddingDim(embeddingDim), hiddenDim(hiddenDim), contextLength(contextLength) {
    initializeWeights();
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::initializeWeights() {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 0.1);

    embeddingMatrix = Eigen::MatrixXd::Zero(vocabSize, embeddingDim);
    hiddenWeights = Eigen::MatrixXd::Zero(embeddingDim * contextLength, hiddenDim);
    hiddenBias = Eigen::VectorXd::Zero(hiddenDim);
    outputWeights = Eigen::MatrixXd::Zero(hiddenDim, vocabSize);
    outputBias = Eigen::VectorXd::Zero(vocabSize);

    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingDim; j++) {
            embeddingMatrix(i, j) = dist(gen);
        }
    }

    for (int i = 0; i < embeddingDim * contextLength; i++) {
        for (int j = 0; j < hiddenDim; j++) {
            hiddenWeights(i, j) = dist(gen);
        }
    }

    for (int i = 0; i < hiddenDim; i++) {
        hiddenBias(i) = dist(gen);
        for (int j = 0; j < vocabSize; j++) {
            outputWeights(i, j) = dist(gen);
        }
    }

    for (int i = 0; i < vocabSize; i++) {
        outputBias(i) = dist(gen);
    }

    embeddingGradients = Eigen::MatrixXd::Zero(vocabSize, embeddingDim);
    hiddenWeightsGradients = Eigen::MatrixXd::Zero(embeddingDim * contextLength, hiddenDim);
    hiddenBiasGradients = Eigen::VectorXd::Zero(hiddenDim);
    outputWeightsGradients = Eigen::MatrixXd::Zero(hiddenDim, vocabSize);
    outputBiasGradients = Eigen::VectorXd::Zero(vocabSize);
}

Eigen::VectorXd NeuralNetwork::forward(const vector<int>& inputTokens) {
    int actualContextLength = min((int)inputTokens.size(), contextLength);

    embeddings = Eigen::VectorXd::Zero(embeddingDim * contextLength);

    for (int i = 0; i < actualContextLength; i++) {
        if (inputTokens[i] < vocabSize && inputTokens[i] >= 0) {
            embeddings.segment(i * embeddingDim, embeddingDim) = embeddingMatrix.row(inputTokens[i]);
        }
    }

    Eigen::VectorXd hiddenInput = hiddenWeights.transpose() * embeddings + hiddenBias;
    hiddenActivations = relu(hiddenInput);

    Eigen::VectorXd output = outputWeights.transpose() * hiddenActivations + outputBias;
    return softmax(output);
}

void NeuralNetwork::backward(const Eigen::VectorXd& prediction, const vector<int>& target) {
    embeddingGradients.setZero();
    hiddenWeightsGradients.setZero();
    hiddenBiasGradients.setZero();
    outputWeightsGradients.setZero();
    outputBiasGradients.setZero();

    Eigen::VectorXd targetVector = Eigen::VectorXd::Zero(vocabSize);
    for (int token : target) {
        if (token < vocabSize && token >= 0) {
            targetVector(token) = 1.0 / target.size();
        }
    }

    Eigen::VectorXd outputError = prediction - targetVector;

    outputWeightsGradients = hiddenActivations * outputError.transpose();
    outputBiasGradients = outputError;

    Eigen::VectorXd hiddenError = outputWeights * outputError;
    Eigen::VectorXd hiddenGradient = hiddenError.cwiseProduct(reluDerivative(hiddenActivations));

    hiddenWeightsGradients = embeddings * hiddenGradient.transpose();
    hiddenBiasGradients = hiddenGradient;

    Eigen::VectorXd embeddingError = hiddenWeights * hiddenGradient;

    for (int i = 0; i < min((int)target.size(), contextLength); i++) {
        if (target[i] < vocabSize && target[i] >= 0) {
            embeddingGradients.row(target[i]) += embeddingError.segment(i * embeddingDim, embeddingDim).transpose();
        }
    }
}

void NeuralNetwork::updateWeights(double learningRate) {
    embeddingMatrix -= learningRate * embeddingGradients;
    hiddenWeights -= learningRate * hiddenWeightsGradients;
    hiddenBias -= learningRate * hiddenBiasGradients;
    outputWeights -= learningRate * outputWeightsGradients;
    outputBias -= learningRate * outputBiasGradients;
}

void NeuralNetwork::saveModel(const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "Error: Could not open file for saving model." << endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
    file.write(reinterpret_cast<const char*>(&embeddingDim), sizeof(embeddingDim));
    file.write(reinterpret_cast<const char*>(&hiddenDim), sizeof(hiddenDim));
    file.write(reinterpret_cast<const char*>(&contextLength), sizeof(contextLength));

    file.write(reinterpret_cast<const char*>(embeddingMatrix.data()), sizeof(double) * embeddingMatrix.size());
    file.write(reinterpret_cast<const char*>(hiddenWeights.data()), sizeof(double) * hiddenWeights.size());
    file.write(reinterpret_cast<const char*>(hiddenBias.data()), sizeof(double) * hiddenBias.size());
    file.write(reinterpret_cast<const char*>(outputWeights.data()), sizeof(double) * outputWeights.size());
    file.write(reinterpret_cast<const char*>(outputBias.data()), sizeof(double) * outputBias.size());

    file.close();
}

void NeuralNetwork::loadModel(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "Error: Could not open file for loading model." << endl;
        return;
    }

    file.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    file.read(reinterpret_cast<char*>(&embeddingDim), sizeof(embeddingDim));
    file.read(reinterpret_cast<char*>(&hiddenDim), sizeof(hiddenDim));
    file.read(reinterpret_cast<char*>(&contextLength), sizeof(contextLength));

    embeddingMatrix.resize(vocabSize, embeddingDim);
    hiddenWeights.resize(embeddingDim * contextLength, hiddenDim);
    hiddenBias.resize(hiddenDim);
    outputWeights.resize(hiddenDim, vocabSize);
    outputBias.resize(vocabSize);

    file.read(reinterpret_cast<char*>(embeddingMatrix.data()), sizeof(double) * embeddingMatrix.size());
    file.read(reinterpret_cast<char*>(hiddenWeights.data()), sizeof(double) * hiddenWeights.size());
    file.read(reinterpret_cast<char*>(hiddenBias.data()), sizeof(double) * hiddenBias.size());
    file.read(reinterpret_cast<char*>(outputWeights.data()), sizeof(double) * outputWeights.size());
    file.read(reinterpret_cast<char*>(outputBias.data()), sizeof(double) * outputBias.size());

    file.close();

    embeddingGradients = Eigen::MatrixXd::Zero(vocabSize, embeddingDim);
    hiddenWeightsGradients = Eigen::MatrixXd::Zero(embeddingDim * contextLength, hiddenDim);
    hiddenBiasGradients = Eigen::VectorXd::Zero(hiddenDim);
    outputWeightsGradients = Eigen::MatrixXd::Zero(hiddenDim, vocabSize);
    outputBiasGradients = Eigen::VectorXd::Zero(vocabSize);
}

Eigen::VectorXd NeuralNetwork::softmax(const Eigen::VectorXd& input) {
    Eigen::VectorXd shifted = input.array() - input.maxCoeff();
    Eigen::VectorXd exp_values = shifted.array().exp();
    return exp_values / exp_values.sum();
}

Eigen::VectorXd NeuralNetwork::relu(const Eigen::VectorXd& input) {
    return input.cwiseMax(0.0);
}

Eigen::VectorXd NeuralNetwork::reluDerivative(const Eigen::VectorXd& input) {
    return (input.array() > 0.0).cast<double>();
}