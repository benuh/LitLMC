#include <iostream>
#include <string>
#include <vector>
#include "text_processor.h"
#include "tokenizer.h"
#include "neural_network.h"
#include "trainer.h"
#include "inference.h"

using namespace std;

void printMenu() {
    cout << "\n=== LitLM - Literature Language Model ===\n";
    cout << "1. Load text from file\n";
    cout << "2. Enter text manually\n";
    cout << "3. Train model\n";
    cout << "4. Ask question\n";
    cout << "5. Generate text\n";
    cout << "6. Save model\n";
    cout << "7. Load model\n";
    cout << "8. Exit\n";
    cout << "Enter your choice: ";
}

int main() {
    TextProcessor textProcessor;
    Tokenizer tokenizer;
    NeuralNetwork neuralNetwork(1000, 128, 256, 32);
    Trainer trainer(&neuralNetwork, &tokenizer);
    Inference inference(&neuralNetwork, &tokenizer);

    vector<string> loadedTexts;
    bool modelTrained = false;

    cout << "Welcome to LitLM - Literature Language Model in C++!\n";

    int choice;
    while (true) {
        printMenu();
        cin >> choice;
        cin.ignore();

        switch (choice) {
            case 1: {
                cout << "Enter filename: ";
                string filename;
                getline(cin, filename);

                if (textProcessor.loadFromFile(filename)) {
                    cout << "File loaded successfully!\n";
                    loadedTexts.push_back(textProcessor.getRawText());
                } else {
                    cout << "Error loading file.\n";
                }
                break;
            }

            case 2: {
                cout << "Enter your text (press Enter twice to finish):\n";
                string text, line;
                while (getline(cin, line) && !line.empty()) {
                    text += line + "\n";
                }

                textProcessor.addText(text);
                loadedTexts.push_back(text);
                cout << "Text added successfully!\n";
                break;
            }

            case 3: {
                if (loadedTexts.empty()) {
                    cout << "No text loaded. Please load some text first.\n";
                    break;
                }

                cout << "Building vocabulary...\n";
                tokenizer.buildVocabulary(loadedTexts);
                cout << "Vocabulary size: " << tokenizer.getVocabSize() << "\n";

                cout << "Training model...\n";
                trainer.trainOnText(loadedTexts, 10, 0.001);
                modelTrained = true;
                cout << "Model trained successfully!\n";
                break;
            }

            case 4: {
                if (!modelTrained) {
                    cout << "Model not trained yet. Please train the model first.\n";
                    break;
                }

                cout << "Enter your question: ";
                string question;
                getline(cin, question);

                string response = inference.generateResponse(question);
                cout << "Response: " << response << "\n";
                break;
            }

            case 5: {
                if (!modelTrained) {
                    cout << "Model not trained yet. Please train the model first.\n";
                    break;
                }

                cout << "Enter prompt: ";
                string prompt;
                getline(cin, prompt);

                string generated = inference.generateText(prompt);
                cout << "Generated text: " << generated << "\n";
                break;
            }

            case 6: {
                if (!modelTrained) {
                    cout << "Model not trained yet. Nothing to save.\n";
                    break;
                }

                cout << "Enter filename to save model: ";
                string filename;
                getline(cin, filename);

                neuralNetwork.saveModel(filename);
                cout << "Model saved successfully!\n";
                break;
            }

            case 7: {
                cout << "Enter filename to load model: ";
                string filename;
                getline(cin, filename);

                neuralNetwork.loadModel(filename);
                modelTrained = true;
                cout << "Model loaded successfully!\n";
                break;
            }

            case 8: {
                cout << "Thank you for using LitLM!\n";
                return 0;
            }

            default: {
                cout << "Invalid choice. Please try again.\n";
                break;
            }
        }
    }

    return 0;
}