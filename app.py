import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

st.set_page_config(page_title="OmniZyme Predictor", page_icon="🧬", layout="centered")

# 1. Define the exact neural network architecture used during training
class FusionMLP(nn.Module):
    def __init__(self, input_dim=1024):
        super(FusionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

# 2. Load models into server memory safely
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # Load Protein Model (ESM-2)
    prot_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    prot_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device).eval()
    
    # Load Chemical Model (ChemBERTa)
    chem_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chem_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(device).eval()
    
    # Load Custom Fusion Model
    fusion_model = FusionMLP().to(device)
    fusion_model.load_state_dict(torch.load("promiscuity_fusion_model.pth", map_location=device))
    fusion_model.eval()
    
    return prot_tokenizer, prot_model, chem_tokenizer, chem_model, fusion_model, device

st.title("🧬 OmniZyme: Substrate Promiscuity Predictor")
st.markdown("""
Determine if a specific enzyme will catalyze a reaction with a target chemical substrate. 
This multi-modal neural network extracts 640-D biological embeddings via **ESM-2** and 384-D chemical embeddings via **ChemBERTa**, fusing them to predict biocatalytic compatibility.
""")

with st.spinner("Spinning up foundation models (ESM-2 & ChemBERTa)..."):
    prot_tokenizer, prot_model, chem_tokenizer, chem_model, fusion_model, device = load_models()

# UI Inputs
st.subheader("Biological Input")
sequence = st.text_area("Enzyme Amino Acid Sequence:", value="MTHQIVTTQTYRVG...", height=100)

st.subheader("Chemical Input")
smiles = st.text_input("Substrate SMILES String:", value="CC(=O)OCC")

if st.button("Predict Reaction ⚡"):
    if sequence and smiles:
        with st.spinner("Extracting multi-modal features and calculating match..."):
            with torch.no_grad():
                # Extract Protein Features
                p_inputs = prot_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1000).to(device)
                p_emb = prot_model(**p_inputs).last_hidden_state[:, 0, :].numpy()
                
                # Extract Chemical Features
                c_inputs = chem_tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128).to(device)
                c_emb = chem_model(**c_inputs).last_hidden_state[:, 0, :].numpy()
                
                # Fuse and Predict
                x_fused = torch.tensor(np.concatenate((p_emb, c_emb), axis=1), dtype=torch.float32).to(device)
                logit = fusion_model(x_fused)
                prob = torch.sigmoid(logit).item()
                
                # Display Results
                st.markdown("---")
                if prob > 0.5:
                    st.success(f"**Reaction Predicted!** (Confidence: {prob*100:.1f}%)")
                    st.info("The structural geometry of this enzyme is highly likely to catalyze this substrate.")
                else:
                    st.error(f"**No Reaction Predicted.** (Confidence: {(1-prob)*100:.1f}%)")
                    st.warning("This substrate is likely incompatible with the enzyme's active site.")
    else:
        st.warning("Please provide both a sequence and a SMILES string.")