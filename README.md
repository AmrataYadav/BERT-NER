# Named Entity Recognition (NER) with Fine-tuned BERT

This project demonstrates the fine-tuning of a **BERT-base-uncased** model for Named Entity Recognition (NER) on the **CONLL 2003** dataset. The goal was to achieve state-of-the-art results while showcasing expertise in deep learning, natural language processing, and model evaluation.

[Link to the fine-tuned model on Hugging Face](https://huggingface.co/AmrataYadav/bert-base-uncased-finetuned-ner)

---

## Results

The fine-tuned model achieved impressive performance metrics:

- **Accuracy**: 98.65%  
- **Precision**: 93.99%  
- **Recall**: 95.13%  
- **F1-score**: 94.56  

---

## Highlights

### Model Architecture
- **Base model**: BERT-base-uncased, fine-tuned for sequence tagging.

### Dataset
- **CONLL 2003**: A benchmark dataset for NER tasks, containing named entity annotations for four categories:
  - **PER** (person)
  - **ORG** (organization)
  - **LOC** (location)
  - **MISC** (miscellaneous)

### Training Details
- **Optimizer**: Adam with betas=(0.9,0.999) and epsilon=1e-08
- **Learning Rate**: 2e-05
- **Learning Rate Scheduler**: Linear
- **Batch Size**: 32  
- **Epochs**: 10  

---

## Key Features

- **Preprocessing**: 
  - Tokenization with Hugging Face's transformers library.  
  - Handling subword tokens to align labels correctly.  

- **Fine-tuning**: 
  - Custom layers added on top of BERT for NER.  
  - Trained using the cross-entropy loss function.  

- **Evaluation**: 
  - Used metrics such as precision, recall, and F1-score specific to NER tasks.  
  - Excluded "O" tags during evaluation.  

---

## Skills Demonstrated

- Proficiency in leveraging pre-trained transformer models for NLP tasks.  
- Understanding of fine-tuning techniques and optimization strategies.  
- Rigorous model evaluation using NER-specific metrics.  
