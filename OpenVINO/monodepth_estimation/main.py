"""
Author: Alan Pipitone
Description: Monodepth Estimation using OpenVINO (CPU inference).
             Creates a heatmap based on the distance between objects and the camera.
Date: 26/10/2024
Email: alan.pipitone@gmail.com
"""

from pathlib import Path
from time import process_time
import cv2
import numpy as np
import openvino as ov
import requests
import openvino.properties as props


# Function to download a file from a given URL and save it to a specified directory
def download_file(url, filename, directory):
    file_path = Path(directory / filename)

    # Create directory structure if it doesn't exist
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Directory {file_path.parent} created.")

    # Download file if it doesn't already exist
    if not file_path.exists():
        print(f"Downloading {file_path}...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"File {file_path} already exists.")

# Download OpenVINO IR format of the MiDaS model
model_folder = Path("model")
ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

model_xml_path = model_folder / ir_model_name_xml

if __name__ == '__main__':

    # Create cache folder
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)

    # Import and compile the model
    core = ov.Core()
    core.set_property({props.cache_dir: cache_folder})
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)

    # Determine network input shape
    network_input_shape = list(input_key.shape)
    network_image_height, network_image_width = network_input_shape[2:]

    # Define a video capture object
    cam = cv2.VideoCapture(0)

    while True:

        ret, frame = cam.read()

        # Resize image to network's expected input shape
        resized_image = cv2.resize(src=frame, dsize=(network_image_height, network_image_width))

        # Reshape the image to match network input shape NCHW
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        t = process_time()
        # Run inference
        result = compiled_model([input_image])[output_key]
        elapsed_time = process_time() - t

        print("Elapsed time: ", elapsed_time)

        # Remove extra dimension (1,H,W -> H,W)
        result = result.squeeze(0)

        # Normalize result to a range of 0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)

        # Apply a colormap for visualization
        result_image = cv2.applyColorMap(result, cv2.COLORMAP_VIRIDIS)

        # Resize back to the original image shape
        result_image = cv2.resize(result_image, frame.shape[:2][::-1])

        # Display the processed frame
        cv2.imshow('Camera', result_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
