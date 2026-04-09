# CLASSIFICATION-BODY-GESTURES-USING-THE-GNN-LSTM-MODEL-FOR-ENDLEES-RUNNER-GAME-CHARACTER-CONTROLLER-
This repository contains a GNN-LSTM model for body gesture classification, including model files, inference code, and a TCP-based interface for real-time control. The dataset is hosted on Kaggle and linked in this repository.

## Overview

This project aims to classify human body gestures and use them as directional controls in a game environment.

The system combines:
- Graph Neural Networks (GNN) for spatial feature extraction
- Long Short-Term Memory (LSTM) for temporal sequence modeling
- MediaPipe for pose estimation
- TCP communication for real-time interaction with external systems (e.g., game engine)

---

## Repository Structure

- main.py # Interface (real-time gesture detection)
- gesture_server.py # TCP server for sending predictions

- gnn_lstm_classifier.pt # Trained GNN-LSTM model
- scaler_mean.npy # Normalization mean
- scaler_scale.npy # Normalization scale

- requirements.txt
- README.md

## Dataset

The dataset used in this project is available on Kaggle:

https://www.kaggle.com/datasets/daffasesarabbani/gesture-video-dataset-for-jump-classification-1

Dataset consists of raw gesture videos:
- jump_in_place  
- jump_left  
- jump_right  
- looking_down  
- still_pose

## Model Performance

The model was evaluated using a test dataset consisting of 5 gesture classes.

# Classification Report

| Class          | Precision | Recall | F1-Score |
|---------------|----------|--------|----------|
| jump_in_place | 0.93     | 0.95   | 0.94     |
| jump_left     | 1.00     | 0.93   | 0.96     |
| jump_right    | 0.93     | 1.00   | 0.96     |
| looking_down  | 0.95     | 1.00   | 0.98     |
| still_pose    | 0.97     | 0.90   | 0.94     |

---

# Overall Performance

- **Accuracy**: 95%  
- **Macro Avg F1-Score**: 0.95  
- **Weighted Avg F1-Score**: 0.95  

How to Run the System ? 

## IMPORTANT

Before running the system, TCP Server (gesture_server.py) or Interface (python main.py), make sure these files are in the root directory:

- gnn_lstm_classifier.pt
- scaler_mean.npy
- scaler_scale.npy

These files are required for:

- Data normalization
- Model inference

## Citation

If you use this project, please cite:

Oddy Virgantara Putra and others, 15.March (2026) Learning Temporal Graph Representations for Intelligent Control in 3D Endless Runner Games. TEKNIKA.

supporting article citations: 

DOI: https://doi.org/10.34148/teknika.v15i1.1416

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

