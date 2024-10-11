# **Mountain Name Recognition Model**

## **Overview**

This project focuses on recognizing mountain names in text using Named Entity Recognition (NER) techniques. The goal is to identify mountain names in unstructured text data. This model can extract named entities related to mountains and label them accordingly.
Check for **comments** inside Jupyter Notebook Demo
## **Data Collection**

Data for this project was sourced from the [Kaggle dataset](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset/code).

- **Text Content**: This dataset includes sentences that describe experiences, sentiments, or activities related to mountainous regions. The text data was preprocessed to annotate mountain names using **BIO labels**.
  
## **Model Training**

The **BERT (Bidirectional Encoder Representations from Transformers)** model was used for token classification. The training process involved:
- Tokenizing the dataset and aligning the BIO labels with each token.
- Fine-tuning the BERT model on the labeled dataset to optimize its performance for identifying mountain names.

## **Model Inference**

Once the model was trained, it was used to predict mountain names in new text.

- The trained model identifies mountain names in input text.
- The following metrics were used to evaluate the model: **F1-Score**, **Recall**, and **Precision**.

## **Example: Mountain Name Recognition**

Here’s an example of how the model predicts mountain names:

**Input Text**: Could you imagine, in the next month I will try to climb the Denali (Mount McKinley).

**Named Entities Recognized as Mountain Names**:
- **Denali**
- **Mount McKinley**

**Detailed Token Predictions**:

| Token        | Label          |
|--------------|----------------|
| could        | O              |
| you          | O              |
| imagine      | O              |
| ,            | O              |
| in           | O              |
| the          | O              |
| next         | O              |
| month        | O              |
| i            | O              |
| will         | O              |
| try          | O              |
| to           | O              |
| climb        | O              |
| the          | O              |
| **denali**   | **B-MOUNTAIN** |
| (            | O              |
| **mount**    | **B-MOUNTAIN** |
| **mckinley** | **I-MOUNTAIN** |
| )            | O              |
| .            | O              |

For more details and additional examples, refer to the Jupyter notebook included in this repository.

## **Problems**

The problems that this model currently faces may include:
  1. Lack of Input Checking: If a user mistakenly inputs "mount **Evrest**" instead of "**Everest**". The model may misrecognize name.
  2. The model struggles when the input contains only part of the names, such as "**Olympus**" instead of "Mount Olympus." So please use full names, from name list file.

All these problems may fix by using larger dataset, leading to better model accuracy.

## **Usage**

To perform inference with the trained model, use the provided Python scripts and download model weights from [Google Drive](https://drive.google.com/drive/folders/19D-1IhhmtL_Lmz29zbp_1AbGXSjpWh1J?usp=sharing).:
Then Git Clone this repository:
```bash
https://github.com/rostyslavshovak/NER_mountain.git
```
After these steps, you should create folders and put model weights(model, logs, results) into your local repository. 
Then use mode_inference.py, input your path to model folder and tesk on your text:

```bash
python model_training.py
python model_inference.py
```
## **Conclusion**
This project demonstrates how to use NER techniques and neural networks to recognize mountain names in text. Future improvements, such as using a larger dataset, could enhance the accuracy and robustness of the model.
