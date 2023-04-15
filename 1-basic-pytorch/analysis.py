import pandas as pd
import torch
from torchmetrics import ConfusionMatrix
import plotly.figure_factory as ff


csv_filename = "test_result_trained.csv"

csv_file = pd.read_csv(csv_filename)
pred = csv_file["pred"]
label = csv_file["label"]

cfm = ConfusionMatrix(task="multiclass", num_classes=10)

confusion_matrix = cfm(torch.tensor(pred.tolist()), torch.tensor(label.tolist()))

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig = ff.create_annotated_heatmap(confusion_matrix.numpy(), x=labels, y=labels)
fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")

fig.write_image("confusion_matrix.png", format="png", scale=2)