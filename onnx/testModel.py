import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("beatnet_bda.onnx")
input_name = sess.get_inputs()[0].name

# Make sure input is NumPy float32
input_array = np.random.randn(1, 1, 272).astype(np.float32)

outputs = sess.run(None, {input_name: input_array})
print("Output shape:", outputs[0].shape)
