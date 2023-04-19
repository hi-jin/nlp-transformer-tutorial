import numpy as np
import pandas as pd
import torch
import torchmetrics
import plotly.figure_factory as ff


NUM_CLASSES = 7

result = pd.read_csv("test_result.csv")

pred = result["predicted_label"]
label = result["reference_label"]

f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")

# 과학(0), 경제(1), 사회(2), 생활문화(3), 세계(4), 스포츠(5), 정치(6)

label_str_to_id = lambda s: {"과학": 0, "경제": 1, "사회": 2, "생활문화": 3, "세계": 4, "스포츠": 5, "정치": 6}[s]

pred = torch.tensor(list(map(label_str_to_id, pred)))
label = torch.tensor(list(map(label_str_to_id, label)))

f1_score = f1(pred, label)
acc = torch.mean((pred == label).float())

print(f"f1_score : {f1_score}") # 0.91
print(f"acc : {acc}") # 0.92


confusion_func = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES)
confusion_matrix = confusion_func(pred, label)

label = ["IT Science", "Economy", "Society", "Life Culture", "World", "Sports", "Politics"]

fig = ff.create_annotated_heatmap(confusion_matrix.numpy(), label, label)
fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
fig.write_image("confusion_matrix.png", format="png", scale=2)
