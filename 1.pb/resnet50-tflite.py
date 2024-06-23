import os
import shutil
import torch
import timm
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Constants
onnx_path = 'resnet50_no_head.onnx'
saved_model_dir = 'tf_saved_model_dir'
tflite_path = 'resnet50_no_head.tflite'

# Step 1: Get ResNet-50, Remove Head, Save to ONNX
model = timm.create_model('resnet50', pretrained=True)
model.reset_classifier(0)  # Remove the classification layer

# Dummy input for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, onnx_path, 
                  input_names=['input'], output_names=['output'], 
                  opset_version=11)

# Step 2: Convert ONNX to TensorFlow SavedModel

# Remove the existing tf_saved_model_dir if it exists
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

# Convert ONNX to TensorFlow SavedModel
os.system(f"onnx-tf convert -i {onnx_path} -o {saved_model_dir}")

# Step 3: Convert TensorFlow SavedModel to TFLite

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Conversion complete: {tflite_path}")
