# 📰 News Article Classification Using DistilBERT

An NLP project that classifies news articles into four categories — **World, Sports, Business, and Sci/Tech** — using a fine-tuned DistilBERT transformer model. The project includes a Streamlit web application for interactive predictions.

## 🚀 Live Demo

🔗 **Streamlit App:** [Try the News Classifier](YOUR_STREAMLIT_APP_URL)

> Replace `YOUR_STREAMLIT_APP_URL` with your actual public Streamlit deployment link.

## 📌 Project Overview

With the increasing volume of digital news, automatically organizing articles into meaningful categories can make information easier to navigate.

This project uses transfer learning with DistilBERT to understand the text of a news article and predict its most likely category.

The model was fine-tuned on the AG News dataset and integrated into a Streamlit application where users can enter an article and receive a prediction.

## 🎯 Objectives

- Build a multi-class news text classification system.
- Fine-tune a pretrained transformer model using Hugging Face Transformers.
- Evaluate the model using accuracy and macro F1-score.
- Develop an interactive web application using Streamlit.
- Explore the limitations of transformer-based text classification on ambiguous and mixed-topic articles.

## 🗂️ Dataset

**AG News Dataset**

- Dataset: [AG News on Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news)
- Total examples: 127,600 across the original training and test splits.
- Original training split: 120,000 articles.
- Original test split: 7,600 articles.

The original training split was divided into training and validation sets:

| Split | Number of articles |
|---|---:|
| Training | 108,000 |
| Validation | 12,000 |
| Test | 7,600 |
| **Total** | **127,600** |

### News Categories

| Label | Category |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## 🧠 Model and Technologies

### Model

**DistilBERT — `distilbert/distilbert-base-uncased`**

DistilBERT is a smaller, faster transformer model derived from BERT. It uses contextual representations of text to support natural language understanding tasks.

The pretrained model was fine-tuned for four-class sequence classification on AG News.

### Technologies Used

- **Python** — programming language
- **PyTorch** — deep learning framework
- **Hugging Face Transformers** — pretrained model, tokenizer, and training utilities
- **Hugging Face Datasets** — dataset loading and preparation
- **Scikit-learn** — evaluation metrics
- **Streamlit** — interactive web application
- **Kaggle Notebooks** — model training using GPU acceleration

## ⚙️ Methodology

1. **Dataset loading:** Load the AG News dataset.
2. **Data preparation:** Split the original training data into training and validation sets.
3. **Tokenization:** Convert article text into token IDs using the DistilBERT tokenizer, with truncation and a maximum sequence length of 128 tokens.
4. **Model fine-tuning:** Train DistilBERT for four-category classification using Hugging Face Trainer.
5. **Validation:** Monitor validation loss, accuracy, and macro F1-score.
6. **Testing:** Evaluate the selected model on the held-out test split.
7. **Model saving:** Save the trained model, configuration, and tokenizer.
8. **Deployment:** Load the saved model in a Streamlit app to classify user-provided articles.

## 📊 Model Performance

The reported evaluation results from the two-epoch training run were:

| Metric | Test result |
|---|---:|
| Test accuracy | 94.34% |
| Test macro F1-score | 94.34% |
| Test loss | 0.3562 |

These results are from the reported two-epoch test evaluation. They should not be attributed to a later checkpoint unless that checkpoint is separately evaluated on the held-out test set.

### Interpretation

- The model performs well on the AG News test dataset.
- Sports articles were among the strongest-performing categories in the initial confusion matrix.
- Business and Sci/Tech articles showed more confusion than some other categories.
- A high test score does not guarantee correct predictions on every real-world article.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/purple279/News_classification
cd News_classification
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check the model directory

Ensure the saved model files are available locally. The application expects the model at:

```text
saved_model/saved_model/
```

The directory should contain files such as `config.json`, `model.safetensors`, and the tokenizer files.

### 5. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📁 Project Structure

```text
News_classification/
├── app.py
├── requirements.txt
├── README.md
└── saved_model/
    └── saved_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
```

The exact files may vary depending on how the model was saved.

## ⚠️ Limitations and Drawbacks

### 1. Confusion Between Categories

Business and Sci/Tech articles can discuss the same event. For example, an article about a semiconductor company investing in a new chip factory contains both technological and financial information. The model may focus on the technology-related terms and predict Sci/Tech when Business is the intended category.

### 2. Limited Context Length

The application uses a maximum sequence length of 128 tokens. Longer articles are truncated, so important information appearing later in the text may be ignored.

### 3. Dataset Domain Limitations

The model was trained on AG News. Its performance may decrease on social media posts, highly specialized journalism, regional news, financial reports, or writing styles that differ from the training data.

### 4. Only Four Categories

The model supports World, Sports, Business, and Sci/Tech. It cannot directly assign an article to categories such as entertainment, health, education, or politics as separate classes.

### 5. Ambiguous and Multi-topic Articles

Some articles legitimately belong to more than one category. This application predicts one primary category, even when multiple labels could be appropriate.

### 6. Confidence Scores Can Be Misleading

Softmax scores indicate the model's relative preference among its four classes. They are not guaranteed to be calibrated probabilities. A high score does not necessarily mean the prediction is correct.

### 7. No Fact-Checking

The model classifies text; it does not verify whether an article's claims are true, detect misinformation reliably, or establish the credibility of its source.

### 8. Resource Requirements

The saved transformer model requires more storage and memory than a small traditional machine-learning classifier. Loading and inference speed also depend on the hardware and hosting environment.

### 9. Small Custom Test Set

A few manually written examples are useful for identifying weaknesses, but they do not provide a statistically reliable estimate of real-world accuracy. The custom examples tested so far exposed difficulties with mixed-topic articles.

## 🔮 Future Improvements

- Compare DistilBERT with BERT and RoBERTa under the same evaluation setup.
- Analyze per-class precision, recall, F1-score, and the confusion matrix.
- Experiment with longer input sequences for lengthy articles.
- Evaluate confidence calibration and consider an abstention option for uncertain predictions.
- Explore multi-label classification for articles that cover multiple subjects.
- Expand testing to include diverse real-world news sources.
- Optimize model size and inference speed for deployment.

## 👩‍💻 What I Learned

Through this project, I gained practical experience with:

- Transformer-based NLP and transfer learning.
- Text tokenization and sequence classification.
- Fine-tuning DistilBERT using PyTorch and Hugging Face Trainer.
- Evaluating multi-class models using accuracy and macro F1-score.
- Saving and loading trained models for inference.
- Building an interactive machine-learning application using Streamlit.
- Identifying model errors and documenting limitations.

## 📚 References

- [AG News Dataset — Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news)
- [DistilBERT Model — Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 Disclaimer

This project is intended for educational and demonstration purposes. Predictions may be incorrect, especially for ambiguous, mixed-topic, or out-of-domain articles. The model should not be used as the sole basis for editorial, financial, or other high-impact decisions.

---

**Built with Python, PyTorch, Hugging Face Transformers, and Streamlit.**
