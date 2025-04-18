# Part 1 

# SpatialGPT: Vanilla Model + Dropout + Distance-Based Evaluation

> File can be found in here: 2_Vanilla_Spatial_GPT/Vanilla_3_Dropout_SimplifiedModel_Prediction_errors.ipynb

This notebook implements the **vanilla version of SpatialGPT**, a GPT-style transformer model trained to predict sequences of **S2 Cell IDs** for spatial data modeling. 
The model is evaluated using **Haversine distance** between actual and generated trajectories.

---

## What This Notebook Does

✅ Loads pre-tokenized spatial data (S2 level-16)  
✅ Trains a TransformerDecoder model with positional encoding and dropout  
✅ Applies early stopping and checkpoint saving  
✅ Generates spatial trajectories using top-k / top-p sampling  
✅ Evaluates spatial errors using Haversine distance  
✅ Visualizes predictions (lat/lng plots) and error per step  


## Model Overview

Input: S2_ID sequence → [t0, t1, ..., tN]
           ↓
Token + Positional Embeddings
           ↓
TransformerDecoder (4 layers, 4 heads)
           ↓
LayerNorm + Linear → Vocabulary logits


Embedding dim: 128, Layers: 4, Heads: 4, Block size: 64, Dropout: 10%, Vocab size: 103,244 (unique S2 Cell IDs)

## Training Configuration
Optimizer: AdamW

LR Scheduler: CosineAnnealing

Loss: CrossEntropy

Early stopping: patience=5

Checkpoints saved every 5k steps

Sampling enabled for progress monitoring

## Evaluation Metrics
Autoregressive generation for 20 future steps

Haversine distance between actual and predicted coordinates

Plots:
1. Actual vs generated lat/lng paths
2. Haversine error per step

## Key Insights
1. Distance-based loss (Haversine) offers richer insight than token accuracy.
2. Dropout and LR scheduling help stabilize long-sequence prediction.
3. Trajectories are generative, not just classification—closer to sequence modeling in NLP.


# Part 2

# SpatialGPT — A GPT-Style Model for Predicting Spatial Trajectories

> File can be found in here: 3_SpatialGPT_Delta_Embeddings_Dual_Decoder/SptialGPT_Normalize_Token_Frequency.ipynb

**SpatialGPT** is a transformer-based autoregressive model designed to predict **next locations** in a spatial sequence using tokenized S2 Cell IDs. 
Inspired by language models like GPT, this project treats **location prediction** as a sequence modeling problem, making it useful for:

- **Trajectory generation**
- **Hotspot prediction**
- **Mobility modeling**
- **Logistics and delivery path forecasting**


## Key Concepts

- **S2 Cell Tokenization**: Uses [Google S2 geometry](https://s2geometry.io/) to convert (lat, lng) points into unique spatial tokens at level 16 granularity.
- **Autoregressive Learning**: Given a sequence of S2 tokens, predict the next one using self-attention and positional embeddings.
- **Evaluation**: Uses **Haversine distance** to compute error between actual and predicted locations.
- **Sampling**: Implements top-k, top-p sampling and temperature scaling to generate realistic location sequences.


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
