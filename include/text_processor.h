#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

class TextProcessor {
public:
    TextProcessor();
    ~TextProcessor();

    bool loadFromFile(const string& filename);
    void addText(const string& text);
    vector<string> preprocessText(const string& text);
    string getRawText() const;
    void clearText();

private:
    string rawText;
    string toLowerCase(const string& text);
    string removePunctuation(const string& text);
    vector<string> splitIntoSentences(const string& text);
};

#endif