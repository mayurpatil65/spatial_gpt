# Part 1 

# SpatialGPT — A GPT-Style Model for Predicting Spatial Trajectories

**SpatialGPT** is a transformer-based autoregressive model designed to predict **next locations** in a spatial sequence using tokenized S2 Cell IDs. 
Inspired by language models like GPT, this project treats **location prediction** as a sequence modeling problem, making it useful for:

- **Trajectory generation**
- **Hotspot prediction**
- **Mobility modeling**
- **Logistics and delivery path forecasting**

---

## Key Concepts

- **S2 Cell Tokenization**: Uses [Google S2 geometry](https://s2geometry.io/) to convert (lat, lng) points into unique spatial tokens at level 16 granularity.
- **Autoregressive Learning**: Given a sequence of S2 tokens, predict the next one using self-attention and positional embeddings.
- **Evaluation**: Uses **Haversine distance** to compute error between actual and predicted locations.
- **Sampling**: Implements top-k, top-p sampling and temperature scaling to generate realistic location sequences.

---

## 🔧 Model Architecture

```text
Input: [S2_ID_t0, S2_ID_t1, ..., S2_ID_tN]
           │
     Token Embedding + Positional Encoding
           │
  ┌───────────────────────────────────────┐
  │ Transformer Decoder (4 layers, 4 heads) │
  └───────────────────────────────────────┘
           │
     LayerNorm → Linear → Vocabulary logits
           │
      Next Token Prediction
