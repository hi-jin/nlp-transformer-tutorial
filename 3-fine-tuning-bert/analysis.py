import numpy as np
import pandas as pd
import torch
import torchmetrics
import plotly.figure_factory as ff

### configs

NUM_CLASSES = 7
csv_filename = "test_result_trained.csv"
csv_file = pd.read_csv(csv_filename)

###

pred = csv_file["pred"]
label = csv_file["label"]

f1_func = torchmetrics.F1Score(
    task="multiclass",
    num_classes=NUM_CLASSES,
)

acc = np.mean(pred == label)
f1 = f1_func(torch.tensor(pred), torch.tensor(label))

print(f"acc : {acc}\tf1 score : {f1}")

confusion_func = torchmetrics.ConfusionMatrix(
    task="multiclass",
    num_classes=NUM_CLASSES,
)

confusion_matrix = confusion_func(torch.tensor(pred), torch.tensor(label))
labels = ["IT Science", "Economy", "Society", "Life Culture", "World", "Sports", "Politics"]

fig = ff.create_annotated_heatmap(confusion_matrix.numpy(), x=labels, y=labels)
fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")

fig.write_image("confusion_matrix.png", format="png", scale=2)
