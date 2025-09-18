#include "tokenizer.h"
#include <sstream>
#include <algorithm>
#include <iostream>

using namespace std;

Tokenizer::Tokenizer() : nextTokenId(0) {
    vocabToId["<UNK>"] = nextTokenId++;
    idToVocab[0] = "<UNK>";

    vocabToId["<PAD>"] = nextTokenId++;
    idToVocab[1] = "<PAD>";

    vocabToId["<START>"] = nextTokenId++;
    idToVocab[2] = "<START>";

    vocabToId["<END>"] = nextTokenId++;
    idToVocab[3] = "<END>";
}

Tokenizer::~Tokenizer() {
}

void Tokenizer::buildVocabulary(const vector<string>& texts) {
    unordered_map<string, int> wordCount;

    for (const string& text : texts) {
        vector<string> words = splitWords(text);
        for (const string& word : words) {
            wordCount[word]++;
        }
    }

    for (const auto& pair : wordCount) {
        if (vocabToId.find(pair.first) == vocabToId.end()) {
            vocabToId[pair.first] = nextTokenId;
            idToVocab[nextTokenId] = pair.first;
            nextTokenId++;
        }
    }

    cout << "Vocabulary built with " << vocabToId.size() << " unique tokens." << endl;
}

vector<int> Tokenizer::tokenize(const string& text) {
    vector<string> words = splitWords(text);
    vector<int> tokens;

    for (const string& word : words) {
        if (vocabToId.find(word) != vocabToId.end()) {
            tokens.push_back(vocabToId[word]);
        } else {
            tokens.push_back(vocabToId["<UNK>"]);
        }
    }

    return tokens;
}

string Tokenizer::detokenize(const vector<int>& tokens) {
    string result;
    for (int i = 0; i < tokens.size(); i++) {
        if (idToVocab.find(tokens[i]) != idToVocab.end()) {
            if (i > 0) result += " ";
            result += idToVocab[tokens[i]];
        }
    }
    return result;
}

int Tokenizer::getVocabSize() const {
    return vocabToId.size();
}

int Tokenizer::getTokenId(const string& token) const {
    auto it = vocabToId.find(token);
    return (it != vocabToId.end()) ? it->second : vocabToId.at("<UNK>");
}

string Tokenizer::getToken(int tokenId) const {
    auto it = idToVocab.find(tokenId);
    return (it != idToVocab.end()) ? it->second : "<UNK>";
}

vector<string> Tokenizer::splitWords(const string& text) {
    vector<string> words;
    istringstream iss(text);
    string word;

    while (iss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);

        string cleanWord;
        for (char c : word) {
            if (isalnum(c)) {
                cleanWord += c;
            }
        }

        if (!cleanWord.empty()) {
            words.push_back(cleanWord);
        }
    }

    return words;
}