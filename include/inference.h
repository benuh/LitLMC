#ifndef INFERENCE_H
#define INFERENCE_H

#include "neural_network.h"
#include "tokenizer.h"
#include <string>
#include <vector>

using namespace std;

class Inference {
public:
    Inference(NeuralNetwork* network, Tokenizer* tokenizer);
    ~Inference();

    string generateResponse(const string& question, int maxTokens = 100);
    string generateText(const string& prompt, int maxTokens = 50);
    double calculateSimilarity(const string& text1, const string& text2);

private:
    NeuralNetwork* neuralNetwork;
    Tokenizer* tokenizer;
    int contextLength;

    vector<int> generateNextTokens(const vector<int>& context, int numTokens);
    int sampleFromProbabilities(const Eigen::VectorXd& probabilities);
};

#endif