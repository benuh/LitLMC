#include "trainer.h"
#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

Trainer::Trainer(NeuralNetwork* network, Tokenizer* tokenizer)
    : neuralNetwork(network), tokenizer(tokenizer), contextLength(32) {
}

Trainer::~Trainer() {
}

void Trainer::trainOnText(const vector<string>& texts, int epochs, double learningRate) {
    auto trainingPairs = createTrainingPairs(texts);

    if (trainingPairs.empty()) {
        cout << "No training data available!" << endl;
        return;
    }

    cout << "Training on " << trainingPairs.size() << " samples for " << epochs << " epochs..." << endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffleTrainingData(trainingPairs);

        double totalLoss = 0.0;
        int batchSize = min(32, (int)trainingPairs.size());

        for (int i = 0; i < trainingPairs.size(); i += batchSize) {
            for (int j = i; j < min(i + batchSize, (int)trainingPairs.size()); j++) {
                const auto& pair = trainingPairs[j];
                const vector<int>& input = pair.first;
                const vector<int>& target = pair.second;

                auto prediction = neuralNetwork->forward(input);
                neuralNetwork->backward(prediction, target);

                double loss = 0.0;
                for (int k = 0; k < target.size() && k < prediction.size(); k++) {
                    if (target[k] < prediction.size()) {
                        loss -= log(max(prediction[target[k]], 1e-15));
                    }
                }
                totalLoss += loss / target.size();
            }

            neuralNetwork->updateWeights(learningRate);
        }

        double avgLoss = totalLoss / trainingPairs.size();
        cout << "Epoch " << (epoch + 1) << "/" << epochs << " - Loss: " << avgLoss << endl;

        if (epoch > 0 && epoch % 5 == 0) {
            learningRate *= 0.95;
        }
    }

    cout << "Training completed!" << endl;
}

double Trainer::calculateLoss(const vector<string>& texts) {
    auto trainingPairs = createTrainingPairs(texts);

    if (trainingPairs.empty()) {
        return 0.0;
    }

    double totalLoss = 0.0;

    for (const auto& pair : trainingPairs) {
        const vector<int>& input = pair.first;
        const vector<int>& target = pair.second;

        auto prediction = neuralNetwork->forward(input);

        double loss = 0.0;
        for (int i = 0; i < target.size() && i < prediction.size(); i++) {
            if (target[i] < prediction.size()) {
                loss -= log(max(prediction[target[i]], 1e-15));
            }
        }
        totalLoss += loss / target.size();
    }

    return totalLoss / trainingPairs.size();
}

void Trainer::setContextLength(int length) {
    contextLength = length;
}

vector<pair<vector<int>, vector<int>>> Trainer::createTrainingPairs(const vector<string>& texts) {
    vector<pair<vector<int>, vector<int>>> trainingPairs;

    for (const string& text : texts) {
        vector<int> tokens = tokenizer->tokenize(text);

        if (tokens.size() < 2) continue;

        for (int i = 0; i <= (int)tokens.size() - contextLength - 1; i++) {
            vector<int> context;
            vector<int> target;

            for (int j = 0; j < contextLength && i + j < tokens.size(); j++) {
                context.push_back(tokens[i + j]);
            }

            if (i + contextLength < tokens.size()) {
                target.push_back(tokens[i + contextLength]);
            }

            if (!context.empty() && !target.empty()) {
                trainingPairs.emplace_back(context, target);
            }
        }

        for (int windowSize = 3; windowSize <= min(10, (int)tokens.size()); windowSize++) {
            for (int i = 0; i <= (int)tokens.size() - windowSize; i++) {
                vector<int> context;
                vector<int> target;

                for (int j = 0; j < windowSize - 1; j++) {
                    context.push_back(tokens[i + j]);
                }

                target.push_back(tokens[i + windowSize - 1]);

                if (!context.empty() && !target.empty()) {
                    trainingPairs.emplace_back(context, target);
                }
            }
        }
    }

    return trainingPairs;
}

void Trainer::shuffleTrainingData(vector<pair<vector<int>, vector<int>>>& data) {
    random_device rd;
    mt19937 gen(rd());
    shuffle(data.begin(), data.end(), gen);
}