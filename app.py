
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./saved_model"

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

st.title("📰 AI News Article Classifier")
st.write(
    "Classify a news article into World, Sports, "
    "Business, or Sci/Tech using fine-tuned DistilBERT."
)

article = st.text_area(
    "Paste your news article",
    height=220,
    placeholder="Enter the full news article here..."
)

if st.button("Classify Article", type="primary", use_container_width=True):
    if not article.strip():
        st.warning("Please enter a news article first.")
    else:
        try:
            with st.spinner("Analyzing your article..."):
                tokenizer, model = load_model()

                inputs = tokenizer(
                    article,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                inputs.pop("token_type_ids", None)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)[0]

                predicted_id = int(probabilities.argmax())
                labels = model.config.id2label
                predicted_category = labels.get(
                    predicted_id,
                    labels.get(str(predicted_id), str(predicted_id))
                )

            st.success(f"Predicted category: {predicted_category}")
            st.subheader("Category scores")

            for i in range(len(probabilities)):
                category = labels.get(i, labels.get(str(i), f"Class {i}"))
                score = probabilities[i].item()

                st.write(f"**{category}: {score:.2%}**")
                st.progress(score)

            st.caption(
                "Scores are model confidence estimates, not guaranteed "
                "probabilities of correctness."
            )

        except Exception as e:
            st.error(f"Unable to classify the article: {e}")
