"""  Purpose: Runs the HAR model inference in real-time on a device like a Raspberry Pi, using a hardware accelerator such as Google Coral Edge TPU or Intel Neural Compute Stick 2 (NCS2) for efficiency. The script is optimized to continuously process incoming radar data frames and output the predicted activity (and optionally the subject’s position via compute_position). It is designed to be lightweight, offloading heavy computation to the accelerator. Key Features:
Model Loading: Depending on the --accelerator argument, it will load the model in the appropriate format:
Google Coral (Edge TPU): Uses the tflite runtime with Edge TPU delegate. This requires you to have a .tflite model file that is compiled for the Edge TPU (usually via the edgetpu_compiler). The script expects a file like model_edgetpu.tflite (or you can adjust the path).
Intel NCS2 (Myriad VPU): Uses OpenVINO Inference Engine to load an IR model (.xml and .bin files). You need to have OpenVINO runtime installed on the device. The script expects model.xml and model.bin files for the network.
CPU (fallback): If no accelerator or unsupported option is given, it can default to running a tflite model on CPU or loading the Keras/PyTorch model directly. (For a Raspberry Pi, a tflite model is recommended even on CPU, for speed).
Data Acquisition: This part would interface with the actual UWB radar sensor to get live data frames. In the code, we put a placeholder for reading new data (get_new_data() or similar). You should integrate your radar’s SDK or data source here. For example, if the radar provides frames via serial or socket, connect and read from it.
Preprocessing: Uses the same preprocessing steps as offline (calls preprocess_sequence) to ensure consistency. In real-time, background subtraction might use a dynamically updated background estimate or a pre-collected “empty room” background.
Inference Loop: Continuously runs inference on incoming data. Typically, it will maintain a buffer of the last N frames (N = sequence length needed by the model) and each time a new frame comes in, it will drop the oldest frame and append the new one, then run the model on the buffer.
Output: Prints or logs the predicted activity. In this script, we also write the latest prediction to a file experiments/results/current_pred.txt so that the dashboard can read and display it. The output can be extended to send data to other systems (MQTT message, etc.).
Usage:
bash
Copy
python real_time_inference.py --accelerator coral
Use --accelerator coral for Google Coral Edge TPU, --accelerator ncs2 for Intel NCS2, or --accelerator cpu for CPU inference. Before running this, you should have:
For Coral: a quantized TFLite model (int8) compiled for Edge TPU (e.g., via edgetpu_compiler).
For NCS2: a model converted to OpenVINO IR format.
For CPU: at least a TFLite model or fallback to the original model (though original model on Pi might be slow).
python
Copy
 """
# code stats from here 

#!/usr/bin/env python3
"""
real_time_inference.py: Run real-time HAR inference on Raspberry Pi with an accelerator (Edge TPU or NCS2).

Continuously reads data from the UWB radar sensor, preprocesses it, and runs the HAR model to detect activities in real time.
Optimized for using hardware accelerators:
    --accelerator coral  (Google Coral Edge TPU)
    --accelerator ncs2   (Intel Neural Compute Stick 2)
    --accelerator cpu    (CPU-only, using tflite or fallback to Keras/PyTorch if needed)

Outputs the current activity prediction (and optionally computes the position using UWB distances).
Writes the latest prediction to a file (experiments/results/current_pred.txt) for integration with the dashboard.

Usage:
    python real_time_inference.py --accelerator coral
"""
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description="Real-time HAR inference using UWB radar and hardware accelerators.")
parser.add_argument("--accelerator", "-a", choices=["coral", "ncs2", "cpu"], default="cpu",
                    help="Accelerator to use: 'coral' for Edge TPU, 'ncs2' for Intel NCS2, 'cpu' for CPU-only.")
parser.add_argument("--model_path", "-m", type=str, default=None,
                    help="Path to the model file (tflite or IR). If not provided, defaults will be used.")
args = parser.parse_args()

# Load class labels for output
label_names = []
try:
    import json
    with open("experiments/results/classes.json", "r") as f:
        label_names = json.load(f)
except FileNotFoundError:
    label_names = []

# Determine model file paths based on accelerator
model_path = args.model_path
interpreter = None
exec_net = None
input_details = output_details = None
openvino_inputs = openvino_outputs = None

if args.accelerator == "coral":
    # Google Coral Edge TPU (use tflite runtime with EdgeTPU delegate)
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        raise ImportError("tflite_runtime not installed. Please install it to use Coral.")
    # Default model path for Coral
    if model_path is None:
        model_path = "experiments/results/model_edgetpu.tflite"
    print(f"Loading TFLite model for Edge TPU: {model_path}")
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
elif args.accelerator == "ncs2":
    # Intel Neural Compute Stick 2 via OpenVINO
    try:
        from openvino.inference_engine import IECore
    except ImportError:
        raise ImportError("OpenVINO not installed or configured. Please install OpenVINO to use NCS2.")
    # Default model path for NCS2 (IR files .xml and .bin)
    if model_path is None:
        model_xml = "experiments/results/model.xml"
        model_bin = "experiments/results/model.bin"
    else:
        model_xml = model_path if model_path.endswith(".xml") else model_path + ".xml"
        model_bin = model_path.replace(".xml", ".bin") if model_path.endswith(".xml") else model_path + ".bin"
    ie = IECore()
    print(f"Loading OpenVINO model for NCS2: {model_xml}")
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="MYRIAD")  # MYRIAD is the device name for Intel NCS2
    openvino_inputs = list(net.input_info.keys())
    openvino_outputs = list(net.outputs.keys())
else:
    # CPU fallback (use tflite without delegate, or could load Keras model directly if small)
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # If tflite_runtime not available, try full TF (could be heavy on Pi)
        from tensorflow.lite.python.interpreter import Interpreter
    # Default CPU tflite model
    if model_path is None:
        model_path = "experiments/results/model.tflite"
    print(f"Loading TFLite model on CPU: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Import preprocessing and position computation
from data_preprocessing import preprocess_sequence
from compute_position import trilateration, anchor_positions  # anchor_positions defined in compute_position.py if needed for position

# Setup for real-time data capture
# Initialize your UWB radar sensor here (pseudocode):
# radar = RadarSensor(port="COM3", baudrate=... )  # example initialization, replace with actual API
sequence_buffer = []  # to store latest frames
sequence_length = None
if interpreter:
    # Get the expected input shape for the model from TFLite interpreter
    seq_shape = input_details[0]['shape']  # e.g., [1, T, H, W, C] or similar
    # Typically shape[0] is 1 (batch), so sequence_length = shape[1]
    if seq_shape.size > 1:
        sequence_length = seq_shape[1]
    else:
        sequence_length = 1
elif exec_net:
    # If using OpenVINO, we might know sequence length from model or define it
    # Assuming sequence length is fixed in model input shape:
    seq_shape = net.input_info[openvino_inputs[0]].input_data.shape  # e.g., [1, T, C, H, W]
    sequence_length = seq_shape[1] if len(seq_shape) > 1 else 1

if sequence_length is None:
    sequence_length = 10  # fallback to some default if cannot determine

print(f"Expected sequence length for model: {sequence_length} frames")

print("Starting real-time inference loop. Press Ctrl+C to exit.")
try:
    while True:
        # 1. Acquire a new radar frame (this is pseudo-code; replace with actual sensor reading)
        # Example: frame = radar.get_frame()  # get a single frame of shape (H, W) or similar
        frame = np.zeros((32, 32))  # placeholder for a radar frame (e.g., 32x32 range-Doppler image)
        # TODO: integrate actual sensor data retrieval here.
        
        # 2. Append frame to buffer and maintain length
        sequence_buffer.append(frame)
        if len(sequence_buffer) > sequence_length:
            sequence_buffer.pop(0)  # remove oldest frame to keep buffer size constant
        
        # If buffer not yet full, wait for more frames
        if len(sequence_buffer) < sequence_length:
            continue
        
        # 3. Preprocess the sequence of frames
        raw_seq = np.array(sequence_buffer)  # shape (T, H, W)
        proc_seq = preprocess_sequence(raw_seq)  # shape (T, H, W, 1)
        # Add batch dimension
        input_data = proc_seq[np.newaxis, ...].astype(np.float32)
        
        # 4. Perform inference using the loaded model
        if interpreter:
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = int(np.argmax(output[0]))
        elif exec_net:
            # For OpenVINO, need to match input key
            infer_request = exec_net.infer(inputs={openvino_inputs[0]: input_data})
            output = infer_request[openvino_outputs[0]]
            pred_idx = int(np.argmax(output[0]))
        else:
            pred_idx = None  # This case shouldn't happen as one of above should be set
        
        # Map prediction to label
        if pred_idx is not None:
            activity_label = label_names[pred_idx] if label_names else str(pred_idx)
        else:
            activity_label = "Unknown"
        
        # 5. (Optional) Compute position if distance measurements available for this frame
        # Here, you would retrieve distances from anchors corresponding to this frame (if sensor provides).
        # For demonstration, we'll skip actual computation or use dummy distances.
        # Example:
        # distances = radar.get_anchor_distances(frame)  # hypothetical method
        # if distances:
        #     position = trilateration(anchor_positions, distances)
        # else:
        #     position = (None, None)
        position = (None, None)
        
        # 6. Output the result (print and/or write to file for dashboard)
        output_msg = f"Activity: {activity_label}"
        if position != (None, None):
            output_msg += f", Position: ({position[0]:.2f}, {position[1]:.2f})"
        print(output_msg)
        # Write current prediction to file (for dashboard or logging)
        with open("experiments/results/current_pred.txt", "w") as f:
            if position != (None, None):
                f.write(f"{activity_label},{position[0]},{position[1]}\n")
            else:
                f.write(f"{activity_label},,\n")
        
        # 7. Small delay (adjust as needed for sensor rate)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Real-time inference stopped by user.")



"""  Comments: Real-time deployment code often needs to be tailored to the specific hardware and environment, so consider this a template:
Integrating Sensor Data: You must replace the placeholder that generates frame = np.zeros((32, 32)) with actual code to get a radar frame. This depends on your radar sensor’s API/SDK. For example, if using a XeThru sensor, you might use their Python API to get the baseband or envelope data.
Model Format: Make sure to convert your trained model to the appropriate format for the accelerator:
For Coral: Use TensorFlow Lite to convert the Keras model (tf.lite.TFLiteConverter), then compile it with Edge TPU Compiler. The resulting .tflite file is what you provide. Ensure the model is quantized to int8 for EdgeTPU support.
For NCS2: Use OpenVINO’s Model Optimizer to convert the Keras model (or a saved .h5 or ONNX exported from PyTorch) to OpenVINO IR format. Provide the .xml and .bin files.
The script’s logic can load a default path, but you can also specify a custom --model_path.
Performance Considerations: The use of hardware accelerators should allow near real-time performance. The exact frame rate supported will depend on model complexity and accelerator throughput. If performance is lagging, consider reducing model size (fewer filters, smaller input dimensions) or sequence length. Additionally, ensure that the Raspberry Pi’s CPU isn’t overloaded by other tasks. Using the Edge TPU or NCS2 will offload the neural network inference from the CPU.
Position Calculation: If you want to integrate position tracking, you need a way to get the distance measurements corresponding to each frame or each moment in time. Some UWB systems might give you direct coordinates or distances. If you have those, you can call the trilateration function as shown. Make sure anchor_positions in compute_position.py matches your real anchor setup.
Output File: The current_pred.txt is overwritten with the latest prediction every iteration. The format is simple CSV (activity, x, y). If position is not computed, it writes empty placeholders for x, y. The dashboard will read this file to display information. Alternatively, you could use a more robust communication (sockets, shared memory, etc.), but a file is simple and effective here.
Graceful Shutdown: The loop runs indefinitely until interrupted. We catch KeyboardInterrupt (Ctrl+C) to exit cleanly. In a deployment, you might run this as a service or daemon script that starts on boot and runs continuously.
This script ensures the model can run continuously on the edge device, making it suitable for live monitoring of activities in a space using UWB radar. """ 
