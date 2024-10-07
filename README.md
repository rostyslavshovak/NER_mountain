#Mountain Name Recognition model

#Overview

    This project focuses on recognizing mountain names in text using Named Entity Recognition (NER) techniques. The goal is to identificate mountain names inside the unstructured text data.

#Data Collection

    Data was used from [Kaggle dataset](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset/code).
    Aboout the text Content: This feature contains the actual text content of each sentence/tweet. It captures the expressions, experiences, or sentiments related to mountainous regions and activities.
    The collected text data was preprocessed to extract and annotate mountain names using BIO labels.

#Model Training

    The BERT (Bidirectional Encoder Representations from Transformers) model was used for token classification.
    The dataset was tokenized, and labels were aligned with tokens.
    The model was then fine-tuned on the labeled dataset.

#Model Inference

    A trained model is used for inference.
    The model predicts mountain names from input text.
    For a more detailed analysis of the model, the following metrics were included: F1-Score, Recall and Precision.
    

#Example: Mountain Name Recognition

    You can more info in Jupyter notebook file.
    Input Text: Could you imagine, in the next month I will try to climb the Denali (Mount McKinley).
    
    **Named Entities Recognized as Mountain Names**:
    - **Denali**
    - **Mount McKinley**

    **Detailed tokens and labels predictions**:

    | Token      | Label          |
    |------------|----------------|
    | could      | O              |
    | you        | O              |
    | imagine    | O              |
    | ,          | O              |
    | in         | O              |
    | the        | O              |
    | next       | O              |
    | month      | O              |
    | i          | O              |
    | will       | O              |
    | try        | O              |
    | to         | O              |
    | climb      | O              |
    | the        | O              |
    | **denali** | **B-MOUNTAIN** |
    | (          | O              |
    | **mount**  | **B-MOUNTAIN** |
    | **mckinley** | **I-MOUNTAIN** |
    | )          | O              |
    | .          | O              |


#Usage

    To perform inference with the trained model, use the provided Python script.
    ```python
    python model_training.py
    python model_inference.py
    ```

#Conclusion

    This project demonstrates the process of recognizing mountain names in text using NER and neural networks. Further improvements, like using larger dataset, can enhance the accuracy and robustness of the model.
