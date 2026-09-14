# LowResourceX

LowResourceX explores cross-lingual knowledge transfer from
English pretrained language models to low-resource NLP systems.

## Initial Target

We initially study English-to-Hindi cross-lingual transfer.

The first experiment will:

1. Fine-tune multilingual BERT on English XNLI.
2. Evaluate the model on English XNLI.
3. Evaluate the same model on Hindi XNLI.
4. Establish the cross-lingual baseline.
5. Introduce an English RoBERTa teacher.
6. Apply knowledge distillation to the multilingual student.
7. Compare baseline and distilled models.

## Planned Extensions

- Low-resource data simulations
- Knowledge Distillation
- Named Entity Recognition
- Aligned-Sequence Knowledge Distillation
- Question Answering
- Upstream Knowledge Transfer
- MLflow experiment tracking
- DVC
- FastAPI
- Streamlit
- Docker
- CI/CD