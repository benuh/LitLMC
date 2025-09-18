#include "text_processor.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cctype>

using namespace std;

TextProcessor::TextProcessor() {
}

TextProcessor::~TextProcessor() {
}

bool TextProcessor::loadFromFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    string line;
    string content;
    while (getline(file, line)) {
        content += line + "\n";
    }

    rawText += content;
    file.close();
    return true;
}

void TextProcessor::addText(const string& text) {
    rawText += text + "\n";
}

vector<string> TextProcessor::preprocessText(const string& text) {
    string cleaned = toLowerCase(text);
    cleaned = removePunctuation(cleaned);
    return splitIntoSentences(cleaned);
}

string TextProcessor::getRawText() const {
    return rawText;
}

void TextProcessor::clearText() {
    rawText.clear();
}

string TextProcessor::toLowerCase(const string& text) {
    string result = text;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string TextProcessor::removePunctuation(const string& text) {
    string result;
    for (char c : text) {
        if (isalnum(c) || isspace(c)) {
            result += c;
        } else {
            result += ' ';
        }
    }
    return result;
}

vector<string> TextProcessor::splitIntoSentences(const string& text) {
    vector<string> sentences;
    istringstream iss(text);
    string sentence;

    while (getline(iss, sentence)) {
        if (!sentence.empty()) {
            sentence.erase(0, sentence.find_first_not_of(" \t"));
            sentence.erase(sentence.find_last_not_of(" \t") + 1);

            if (!sentence.empty()) {
                sentences.push_back(sentence);
            }
        }
    }

    return sentences;
}