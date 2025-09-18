#include "inference.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;

Inference::Inference(NeuralNetwork* network, Tokenizer* tokenizer)
    : neuralNetwork(network), tokenizer(tokenizer), contextLength(32) {
}

Inference::~Inference() {
}

string Inference::generateResponse(const string& question, int maxTokens) {
    vector<int> questionTokens = tokenizer->tokenize(question);

    if (questionTokens.empty()) {
        return "I don't understand the question.";
    }

    vector<int> context = questionTokens;

    if (context.size() > contextLength) {
        context = vector<int>(context.end() - contextLength, context.end());
    }

    vector<int> responseTokens = generateNextTokens(context, maxTokens);

    string response = tokenizer->detokenize(responseTokens);

    if (response.empty()) {
        response = "I need more training data to answer that question.";
    }

    response.erase(0, response.find_first_not_of(" \t"));
    response.erase(response.find_last_not_of(" \t") + 1);

    return response;
}

string Inference::generateText(const string& prompt, int maxTokens) {
    vector<int> promptTokens = tokenizer->tokenize(prompt);

    if (promptTokens.empty()) {
        promptTokens.push_back(tokenizer->getTokenId("<START>"));
    }

    vector<int> context = promptTokens;

    if (context.size() > contextLength) {
        context = vector<int>(context.end() - contextLength, context.end());
    }

    vector<int> generatedTokens = generateNextTokens(context, maxTokens);

    vector<int> fullResponse = promptTokens;
    fullResponse.insert(fullResponse.end(), generatedTokens.begin(), generatedTokens.end());

    return tokenizer->detokenize(fullResponse);
}

double Inference::calculateSimilarity(const string& text1, const string& text2) {
    vector<int> tokens1 = tokenizer->tokenize(text1);
    vector<int> tokens2 = tokenizer->tokenize(text2);

    if (tokens1.empty() || tokens2.empty()) {
        return 0.0;
    }

    auto pred1 = neuralNetwork->forward(tokens1);
    auto pred2 = neuralNetwork->forward(tokens2);

    if (pred1.size() != pred2.size()) {
        return 0.0;
    }

    double dotProduct = pred1.dot(pred2);
    double norm1 = pred1.norm();
    double norm2 = pred2.norm();

    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }

    return dotProduct / (norm1 * norm2);
}

vector<int> Inference::generateNextTokens(const vector<int>& context, int numTokens) {
    vector<int> result;
    vector<int> currentContext = context;

    for (int i = 0; i < numTokens; i++) {
        if (currentContext.size() > contextLength) {
            currentContext = vector<int>(currentContext.end() - contextLength, currentContext.end());
        }

        auto probabilities = neuralNetwork->forward(currentContext);

        if (probabilities.size() == 0) {
            break;
        }

        int nextToken = sampleFromProbabilities(probabilities);

        if (nextToken == tokenizer->getTokenId("<END>") || nextToken == tokenizer->getTokenId("<PAD>")) {
            break;
        }

        result.push_back(nextToken);
        currentContext.push_back(nextToken);

        if (result.size() >= 3) {
            bool hasRepeatingPattern = true;
            for (int j = 1; j <= min(3, (int)result.size() / 2); j++) {
                if (result.size() < 2 * j) continue;
                hasRepeatingPattern = true;
                for (int k = 0; k < j; k++) {
                    if (result[result.size() - 1 - k] != result[result.size() - 1 - j - k]) {
                        hasRepeatingPattern = false;
                        break;
                    }
                }
                if (hasRepeatingPattern) break;
            }
            if (hasRepeatingPattern) break;
        }
    }

    return result;
}

int Inference::sampleFromProbabilities(const Eigen::VectorXd& probabilities) {
    random_device rd;
    mt19937 gen(rd());

    double temperature = 0.8;
    Eigen::VectorXd adjustedProbs = probabilities.array().pow(1.0 / temperature);
    adjustedProbs = adjustedProbs / adjustedProbs.sum();

    double topP = 0.9;
    vector<pair<double, int>> probIndexPairs;
    for (int i = 0; i < adjustedProbs.size(); i++) {
        probIndexPairs.emplace_back(adjustedProbs[i], i);
    }
    sort(probIndexPairs.rbegin(), probIndexPairs.rend());

    double cumulativeProb = 0.0;
    vector<pair<double, int>> filteredPairs;
    for (const auto& pair : probIndexPairs) {
        cumulativeProb += pair.first;
        filteredPairs.push_back(pair);
        if (cumulativeProb >= topP) break;
    }

    if (filteredPairs.empty()) {
        filteredPairs = probIndexPairs;
    }

    double totalProb = 0.0;
    for (const auto& pair : filteredPairs) {
        totalProb += pair.first;
    }

    uniform_real_distribution<double> dist(0.0, totalProb);
    double randomValue = dist(gen);

    double currentSum = 0.0;
    for (const auto& pair : filteredPairs) {
        currentSum += pair.first;
        if (randomValue <= currentSum) {
            return pair.second;
        }
    }

    return filteredPairs.empty() ? 0 : filteredPairs.back().second;
}