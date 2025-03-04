# Bangla Hate Speech Detection

This project implements a hate speech detection system for Bangla text using a benchmark dataset and the Bangla-BERT model. The pipeline covers data preprocessing, model fine-tuning, training, evaluation, visualization of training progress, and saving/loading the final model.

## Overview

The primary goal is to accurately detect hate speech in Bangla language text. This project leverages the [Bangla-BERT](https://huggingface.co/sagorsarker/bangla-bert-base) model from Hugging Face, fine-tuned for sequence classification using a benchmark dataset in JSON format.

## Hosted Model

A fine-tuned version of this model is hosted on Hugging Face. You can check it out [here]([https://huggingface.co/your-hf-model](https://huggingface.co/spaces/Kxngh/hatespeechBangla)).  


## Features

- **Data Preprocessing**
  - Loads training and test data from `train.json` and `test.json`.
  - Maps text labels (e.g., `"N"` for non-offensive and `"O"` for offensive) to numeric values.
  - Tokenizes text using the Bangla-BERT tokenizer with proper padding and truncation.

- **Model Training**
  - Fine-tunes the Bangla-BERT model over 3 epochs using PyTorch.
  - Utilizes the AdamW optimizer and CrossEntropy loss.
  - Evaluates the model after each epoch, printing metrics such as Accuracy, Precision, Recall, and F1 Score.

- **Visualization**
  - Plots training loss and evaluation metrics using matplotlib.

- **Model Persistence**
  - Saves the trained model and tokenizer using Hugging Face's `save_pretrained` method.
  - Demonstrates model saving and loading using Python's `pickle` module.

## Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies:**
  ```bash
pip install torch transformers scikit-learn pandas numpy matplotlib tqdm
  ```

## Data
The dataset used for this project is this [Hasoc 2024 Bangla DS](https://hasocfire.github.io/hasoc/2024/dataset.html).
Place your dataset files in the project directory:

- **train.json** – Contains the training samples.
- **test.json** – Contains the test samples.

Each JSON entry should include:

- `text`: The Bangla text sample.
- `offensive_gold`: The label for hate speech (e.g., `"N"` for non-offensive, `"O"` for offensive).

The project maps these labels using:

```python
label_mapping = {"N": 0, "O": 1}
```

##Contributing
Contributions to improve the project are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

##License
This project is licensed under the MIT License.
