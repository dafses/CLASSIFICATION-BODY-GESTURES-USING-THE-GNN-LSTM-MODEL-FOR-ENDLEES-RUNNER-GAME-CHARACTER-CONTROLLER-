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

How to Run the System ? 

##IMPORTANT

Before running the system, TCP Server (gesture_server.py) or Interface (python main.py), make sure these files are in the root directory:

- gnn_lstm_classifier.pt
- scaler_mean.npy
- scaler_scale.npy

These files are required for:

- Data normalization
- Model inference

## Citation

If you use this project, please cite:

Classification of the Direction of the Game Character Controller Based on Body Gestures Using GNN-LSTM

supporting article citations: 

DOI: https://doi.org/10.34148/teknika.v15i1.1416

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

