import cv2
import time
import torch
import mediapipe as mp
import numpy as np
from collections import deque
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool
import torch.nn as nn

# ================= CONFIG =================
MAX_LEN = 50
NUM_KEYPOINTS = 33
KEYPOINT_DIM = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "jump_in_place",
    "jump_left",
    "jump_right",
    "looking_down",
    "still_pose"
]
id2label = {i: l for i, l in enumerate(LABELS)}

# ================= LOAD SCALER =================
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

def normalize_keypoints(kp):
    return (kp - scaler_mean) / scaler_scale

# ================= GRAPH EDGE =================
mediapipe_edges = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),
    (12,14),(14,16),(16,18),
    (11,23),(12,24),(23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]
edge_index = torch.tensor(
    mediapipe_edges, dtype=torch.long
).t().contiguous().to(DEVICE)

# ================= MODEL =================
class GNNLSTMClassifier(nn.Module):
    def __init__(self, node_feat_dim=3, gnn_hidden=64, lstm_hidden=128, num_classes=5):
        super().__init__()
        self.gnn1 = GraphConv(node_feat_dim, gnn_hidden)
        self.gnn2 = GraphConv(gnn_hidden, gnn_hidden)
        self.lstm = nn.LSTM(gnn_hidden, lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, graph_seq):
        embeds = []

        for g in graph_seq:
            x = torch.relu(self.gnn1(g.x, g.edge_index))
            x = torch.relu(self.gnn2(x, g.edge_index))
            x = global_mean_pool(
                x,
                torch.zeros(x.size(0), dtype=torch.long).to(x.device)
            )
            embeds.append(x.squeeze(0))  # [F]

        seq = torch.stack(embeds).unsqueeze(0)  # [1, T, F]
        out, _ = self.lstm(seq)
        return self.classifier(out[:, -1, :])

# ================= LOAD MODEL =================
model = GNNLSTMClassifier(num_classes=len(LABELS)).to(DEVICE)
model.load_state_dict(
    torch.load("gnn_lstm_classifier.pt", map_location=DEVICE)
)
model.eval()

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# ================= BUFFER =================
frame_buffer = deque(maxlen=MAX_LEN)
window_count = 0

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("🚀 Real-time gesture recognition running...")

while cap.isOpened():
    start_total = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        keypoints = np.array([
            [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
        ])
        keypoints = normalize_keypoints(keypoints)
        frame_buffer.append(keypoints)

    prediction = "WAITING"
    inference_ms = 0.0

    # ================= WINDOW-BASED INFERENCE =================
    if len(frame_buffer) == MAX_LEN:
        start_inf = time.time()
        graph_seq = []

        for kp in frame_buffer:
            graph_seq.append(
                Data(
                    x=torch.tensor(kp, dtype=torch.float).to(DEVICE),
                    edge_index=edge_index
                )
            )

        with torch.no_grad():
            out = model(graph_seq)
            pred_id = out.argmax(dim=1).item()
            prediction = id2label[pred_id]

        inference_ms = (time.time() - start_inf) * 1000
        window_count += 1

        total_delay = (time.time() - start_total) * 1000
        fps = 1 / (time.time() - prev_time)

        # ===== LOG KE CONSOLE (INI YANG LO MAU) =====
        print(
            f"[WINDOW {window_count}] "
            f"Gesture: {prediction} | "
            f"FPS: {fps:.2f} | "
            f"Inference: {inference_ms:.2f} ms | "
            f"Total Delay: {total_delay:.2f} ms"
        )

        # 🔥 RESET BUFFER
        frame_buffer.clear()

    # ================= METRIC =================
    total_delay = (time.time() - start_total) * 1000
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    # ================= OVERLAY =================
    cv2.putText(frame, f"Gesture: {prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"Inference: {inference_ms:.2f} ms", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Total Delay: {total_delay:.2f} ms", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{MAX_LEN}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    cv2.putText(frame, f"Window: {window_count}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    cv2.imshow("GNN-LSTM Gesture Interface", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
