# LitLM - Literature Language Model in C++

A lightweight C++ implementation of a literature-focused language model that can learn from text inputs and answer questions about the content.

## Features

- **Text Input**: Load literature from files or input text manually
- **Neural Network**: Custom feedforward neural network with embeddings
- **Tokenization**: Word-level tokenization with vocabulary building
- **Training**: Train the model on your literature corpus
- **Inference**: Ask questions and generate text based on learned content
- **Model Persistence**: Save and load trained models

## Dependencies

- CMake (3.16 or higher)
- Eigen3 library
- C++17 compatible compiler

## Installation

### Install Eigen3

**macOS (using Homebrew):**
```bash
brew install eigen
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libeigen3-dev
```

### Build the Project

**Option 1: Using the build script (Recommended)**
```bash
./build.sh
```

**Option 2: Manual build**
```bash
mkdir build
cd build
cmake ..
make
```

## How to Run

### Quick Start

1. **Build the project:**
   ```bash
   ./build.sh
   ```

2. **Run the application:**
   ```bash
   cd build
   ./LitLM
   ```

3. **Try the sample data:**
   - Choose option 1 (Load text from file)
   - Enter: `../data/sample.txt`
   - Choose option 3 (Train model) and wait for training to complete
   - Choose option 4 (Ask question) and try asking about Gatsby!

### Menu Options

When you run the program, you'll see an interactive menu with these options:

1. **Load text from file**: Import literature from a text file
2. **Enter text manually**: Input text directly into the system
3. **Train model**: Build vocabulary and train the neural network
4. **Ask question**: Query the trained model about the literature
5. **Generate text**: Generate new text based on a prompt
6. **Save model**: Save the trained model to disk
7. **Load model**: Load a previously saved model
8. **Exit**: Close the application

### Complete Example Workflow

```bash
# 1. Build and run
./build.sh
cd build
./LitLM

# 2. In the program menu:
# Choose: 1 (Load text from file)
# Enter filename: ../data/sample.txt
# You should see: "File loaded successfully!"

# 3. Train the model:
# Choose: 3 (Train model)
# Wait for: "Building vocabulary..." and "Training model..."
# You should see: "Model trained successfully!"

# 4. Ask questions:
# Choose: 4 (Ask question)
# Try questions like:
#   - "Who is Gatsby?"
#   - "What advice did the father give?"
#   - "What is the green light?"

# 5. Generate text:
# Choose: 5 (Generate text)
# Try prompts like:
#   - "In my younger years"
#   - "Gatsby was"

# 6. Save your trained model:
# Choose: 6 (Save model)
# Enter filename: my_model.bin
```

### Tips for Best Results

- **Use longer texts**: The model works better with more training data
- **Be patient**: Training may take a few minutes depending on text size
- **Ask specific questions**: More specific questions often yield better responses
- **Experiment with prompts**: Try different starting phrases for text generation

## Architecture

- **TextProcessor**: Handles file I/O and text preprocessing
- **Tokenizer**: Converts text to numerical tokens and manages vocabulary
- **NeuralNetwork**: Feedforward network with embedding layer, hidden layer, and output layer
- **Trainer**: Manages the training process with backpropagation
- **Inference**: Generates responses and text using the trained model

## Model Details

- **Vocabulary Size**: Up to 1000 tokens (configurable)
- **Embedding Dimension**: 128
- **Hidden Layer Size**: 256
- **Context Length**: 32 tokens
- **Activation**: ReLU for hidden layer, Softmax for output
- **Training**: Stochastic gradient descent with learning rate decay

## Example Files

Create a text file with your literature content:

```
literature.txt
--------------
In the beginning was the Word, and the Word was with God, and the Word was God.
The quick brown fox jumps over the lazy dog.
To be or not to be, that is the question.
```

Then load this file in the program and train the model to start asking questions!

## Contributing

Feel free to contribute improvements, bug fixes, or new features to this literature language model project.

## License

This project is open source and available under the MIT License.