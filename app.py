import io
import os
import tempfile
import cv2
import imageio
import shutil
from flask import Flask, request, render_template, jsonify
import base64
from video_processor import process_video_with_tracknet

app = Flask(__name__)

# Global variables to store video frames
input_frames = []
fps = 0

# Directory to save processed videos
OUTPUT_DIR = "/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/video_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global input_frames, fps
    video_file = request.files['video']
    video_bytes = video_file.read()

    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    # Read video frames
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        return jsonify({'error': 'Could not open video file'}), 400

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()

    if not frames:
        os.remove(temp_path)
        return jsonify({'error': 'No frames found in video'}), 400

    input_frames = frames

    # Return the video as base64 for display
    video_file.seek(0)
    video_b64 = base64.b64encode(video_file.read()).decode('utf-8')

    # Clean up temporary file
    os.remove(temp_path)

    return jsonify({'video': video_b64})


@app.route('/process', methods=['POST'])
def process():
    global input_frames, fps
    if not input_frames:
        return jsonify({'error': 'No video uploaded'}), 400

    # Process the video using TrackNet
    processed_frames = process_video_with_tracknet(input_frames)

    # Convert processed frames to video using imageio with H.264 codec
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_path = temp_file.name
        # Use imageio to write video with H.264 codec
        writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', macro_block_size=1)
        for frame in processed_frames:
            # Convert BGR (OpenCV) to RGB (imageio expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
        writer.close()

    # Save the processed video to OUTPUT_DIR
    output_filename = os.path.join(OUTPUT_DIR, f"processed_video_{int(os.path.getmtime(temp_path))}.mp4")
    shutil.copy(temp_path, output_filename)

    # Read the video for base64 encoding
    with open(temp_path, 'rb') as f:
        output_video = f.read()

    video_b64 = base64.b64encode(output_video).decode('utf-8')

    # Clean up temporary file
    os.remove(temp_path)

    return jsonify({'video': video_b64})


if __name__ == "__main__":
    app.run(debug=True)