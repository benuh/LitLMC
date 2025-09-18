#ifndef TRAINER_H
#define TRAINER_H

#include "neural_network.h"
#include "tokenizer.h"
#include <vector>
#include <string>

using namespace std;

class Trainer {
public:
    Trainer(NeuralNetwork* network, Tokenizer* tokenizer);
    ~Trainer();

    void trainOnText(const vector<string>& texts, int epochs, double learningRate);
    double calculateLoss(const vector<string>& texts);
    void setContextLength(int length);

private:
    NeuralNetwork* neuralNetwork;
    Tokenizer* tokenizer;
    int contextLength;

    vector<pair<vector<int>, vector<int>>> createTrainingPairs(const vector<string>& texts);
    void shuffleTrainingData(vector<pair<vector<int>, vector<int>>>& data);
};

#endif