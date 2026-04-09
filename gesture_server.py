import socket
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool
import torch.nn as nn

# ===============================
# CONFIG
# ===============================
HOST = "0.0.0.0"
PORT = 5005
SEQ_LEN = 50
NUM_KEYPOINTS = 33

LABELS = ["jump_in_place", "jump_left", "jump_right", "looking_down", "still_pose"]

# ===============================
# LOAD SCALER
# ===============================
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

# ===============================
# MEDIAPIPE
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ===============================
# GRAPH EDGE
# ===============================
edges = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),
    (12,14),(14,16),
    (23,24),(23,25),(25,27),(27,29),
    (24,26),(26,28),(28,30)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ===============================
# MODEL
# ===============================
class GNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = GraphConv(3, 64)
        self.gnn2 = GraphConv(64, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, len(LABELS))

    def forward(self, graph_seq):
        embeddings = []

        for g in graph_seq:
            x = self.gnn1(g.x, g.edge_index)
            x = self.gnn2(x, g.edge_index)
            x = torch.relu(x)
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
            embeddings.append(x)

        seq = torch.stack(embeddings).unsqueeze(0)
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1])

# ===============================
# LOAD MODEL
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNLSTM().to(device)
model.load_state_dict(torch.load("gnn_lstm_classifier.pt", map_location=device))
model.eval()

# ===============================
# TCP SERVER
# ===============================
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("🚀 Server started, waiting for Unity...")

conn, addr = server.accept()
print("✅ Unity connected:", addr)

# ===============================
# MAIN LOOP
# ===============================
buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])

        keypoints = np.array(keypoints)

        # normalize
        keypoints = (keypoints - scaler_mean) / scaler_scale
        buffer.append(keypoints)

    if len(buffer) == SEQ_LEN:
        graph_seq = []
        for frame in buffer:
            g = Data(
                x=torch.tensor(frame, dtype=torch.float),
                edge_index=edge_index
            )
            graph_seq.append(g)

        with torch.no_grad():
            out = model(graph_seq)
            pred = torch.argmax(out, dim=1).item()
            gesture = LABELS[pred]

        print("Predicted:", gesture)
        conn.sendall((gesture + "\n").encode())

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
conn.close()
server.close()
