#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    void buildVocabulary(const vector<string>& texts);
    vector<int> tokenize(const string& text);
    string detokenize(const vector<int>& tokens);
    int getVocabSize() const;
    int getTokenId(const string& token) const;
    string getToken(int tokenId) const;

private:
    unordered_map<string, int> vocabToId;
    unordered_map<int, string> idToVocab;
    int nextTokenId;

    vector<string> splitWords(const string& text);
};

#endif