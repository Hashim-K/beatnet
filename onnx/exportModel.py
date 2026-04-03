import sys
import os
import torch
import torch.nn as nn
import onnx
sys.path.insert(0, os.path.abspath("../src"))
from BeatNet.BeatNet import BeatNet

class BeatNetWithSoftmax(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)
    
model_path = "beatnet_bda.onnx"

# Initialize BeatNet
estimator = BeatNet(1, mode='stream', inference_model='PF', plot=[], thread=False)

# Get the PyTorch model (BDA) + softmax
model = BeatNetWithSoftmax(estimator.model)

# set to eval mode
model.eval()

# move it to the CPU
device = torch.device("cpu")
model.to(device)

# Create dummy input (batch_size, time_steps, feature_dim)
print("Expected dim_in:", model.base_model.dim_in)
dummy_input = torch.randn(1, 1, 272).to(device)

# export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    model_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch", 2: "time"}},
    opset_version=17
)

print(f"Exported to {model_path}")

try:
    onnx.checker.check_model(model_path)
    print("ONNX model is valid.")
except:
    print("Error: ONNX model is not valid...")