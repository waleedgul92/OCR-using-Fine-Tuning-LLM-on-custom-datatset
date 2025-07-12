This project focuses on **Optical Character Recognition (OCR)** using a **Vision-Encoder-Decoder model**, specifically fine-tuning a pre-trained **TrOCR-small-printed** model. The goal is to accurately transcribe text from captcha images.

## Project Structure

The project is structured as a Jupyter Notebook (`OCR_using_LLM.ipynb`) that guides you through the following steps:

1.  **Environment Setup**: Installs necessary libraries like `transformers`, `datasets`, `jiwer`, and `sentencepiece`.
2.  **Data Loading and Preparation**:
      * Mounts Google Drive to access image data.
      * Loads captcha images from a specified directory (`/content/drive/MyDrive/Images/Images/`).
      * Extracts labels (text) from the image filenames.
      * Organizes image filenames and labels into a Pandas DataFrame.
      * Shuffles the dataset.
3.  **Dataset Splitting**: Divides the prepared dataset into training and testing sets using `sklearn.model_selection.train_test_split`.
4.  **Custom Dataset Class (`IAMDataset`)**:
      * Defines a custom PyTorch `Dataset` to handle image loading and preprocessing.
      * Uses `TrOCRProcessor` to prepare images (resize, normalize) and encode text labels.
      * Handles padding and special tokens for the labels.
5.  **Dataloader Creation**: Creates `DataLoader` instances for both training and evaluation datasets to facilitate batch processing.
6.  **Model Initialization**:
      * Loads a pre-trained `TrOCR-small-printed` model from the `transformers` library using `VisionEncoderDecoderModel`.
      * Moves the model to the GPU if available.
7.  **Model Configuration**: Configures essential model attributes for text generation, including:
      * `decoder_start_token_id`
      * `pad_token_id`
      * `vocab_size`
      * Beam-search parameters (`eos_token_id`, `max_length`, `early_stopping`, `no_repeat_ngram_size`, `length_penalty`, `num_beams`).
8.  **Metrics Setup**: Loads the Character Error Rate (CER) metric using the `evaluate` library to quantify transcription accuracy.
9.  **Training Loop**:
      * Implements a training loop for a specified number of epochs (50 in this case).
      * Uses `AdamW` optimizer for model parameter updates.
      * Calculates and prints the training loss after each epoch.
      * Evaluates the model on the validation set after each epoch by generating text and computing the CER.
      * Saves model weights if the validation CER falls below a certain threshold (0.05).
10. **Inference**:
      * Loads the fine-tuned model.
      * Performs inference on a new set of unlabeled captcha images from `/content/drive/MyDrive/unlabel_captcha/`.
      * Prints the predicted label for each image.
      * Stores the predicted labels in a dictionary.
      * Saves the predicted labels to a CSV file (`predicted_labels_new_data.csv`).
11. **Dependency Management**: Generates a `requirements.txt` file listing all installed Python packages.

## Getting Started

### Prerequisites

  * Python 3.x
  * Google Colab (recommended for GPU access) or a local environment with PyTorch and CUDA configured.

### Installation

The notebook handles most of the installations. If running locally, ensure you have `pip` installed and then run:

```bash
pip install -q transformers datasets jiwer sentencepiece evaluate
```

### Data

The project expects captcha images to be organized in a directory structure. Make sure your images are in `/content/drive/MyDrive/Images/Images/` (for training/evaluation) and `/content/drive/MyDrive/unlabel_captcha/` (for inference), or modify the `data_dir` and `path` variables in the notebook accordingly. Image filenames should correspond to their labels (e.g., `87w822.png` for the label `87w822`).

### Running the Notebook

1.  Open the `OCR_using_LLM.ipynb` notebook in Google Colab or Jupyter Notebook.
2.  Run all cells sequentially.
3.  Ensure your Google Drive is mounted correctly when prompted.

## Results

The training process aims to reduce the Character Error Rate (CER) on the validation set. The notebook includes print statements for the loss after each training epoch and the validation CER. If the validation CER drops below 0.05, the model weights are saved.

The inference section demonstrates how to load the saved model and predict labels for new images. The predicted labels are also saved to a CSV file.

## Dependencies

The `requirements.txt` file generated at the end of the notebook lists all the exact versions of the Python packages used in the environment, ensuring reproducibility.

## Acknowledgements

This project utilizes pre-trained models from the Hugging Face Transformers library.
