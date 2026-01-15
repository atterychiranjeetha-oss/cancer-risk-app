import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load model + metadata
# -----------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("rf_model_with_meta.joblib")
    return bundle["model"], bundle["organs"], bundle["max_seq_len"]

model, ORGANS, MAX_SEQ_LEN = load_model()

DNA_FEATURES = MAX_SEQ_LEN * 4
TOTAL_FEATURES = DNA_FEATURES + len(ORGANS)

# -----------------------------
# Encoding functions
# -----------------------------
def encode_sequence(seq):
    seq = seq.upper()
    encoding = []

    for base in seq[:MAX_SEQ_LEN]:
        encoding.extend([
            1 if base == "A" else 0,
            1 if base == "T" else 0,
            1 if base == "C" else 0,
            1 if base == "G" else 0,
        ])

    while len(encoding) < DNA_FEATURES:
        encoding.extend([0, 0, 0, 0])

    return encoding[:DNA_FEATURES]

def encode_organ(organ):
    return [1 if organ == o else 0 for o in ORGANS]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Cancer Risk Prototype", layout="centered")

st.title("🧬 Multi-Organ Cancer Risk Prototype")
st.caption("Educational proof-of-concept using synthetic genomic data")

st.markdown("---")

dna_sequence = st.text_area(
    "Enter DNA sequence (A / T / C / G only)",
    placeholder="ATCGTAGCTAGCTAGCTAGC",
    height=120
)

selected_organ = st.selectbox(
    "Select organ",
    ORGANS
)

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk"):
    if not dna_sequence.strip():
        st.error("Please enter a DNA sequence.")
    else:
        # Encode inputs
        dna_features = encode_sequence(dna_sequence)
        organ_features = encode_organ(selected_organ)

        final_input = np.array([dna_features + organ_features])

        # Safety check
        if final_input.shape[1] != TOTAL_FEATURES:
            st.error(
                f"Feature mismatch: expected {TOTAL_FEATURES}, "
                f"got {final_input.shape[1]}"
            )
        else:
            probability = model.predict_proba(final_input)[0][1]

            st.success(
                f"Estimated cancer risk for **{selected_organ}**: "
                f"**{probability * 100:.2f}%**"
            )

            st.info(
                "⚠️ This is a research prototype trained on synthetic data. "
                "It is NOT a medical diagnostic tool."
            )

st.markdown("---")
st.caption("Built as an ML research prototype • Not for clinical use")
