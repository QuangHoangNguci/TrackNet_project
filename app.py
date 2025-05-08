import io
import os
import tempfile
import cv2
import imageio
import shutil
from flask import Flask, request, render_template, jsonify, Response
import base64
from video_processor import process_video_with_tracknet
import numpy as np

app = Flask(__name__)

# Directory to save processed videos
OUTPUT_DIR = "/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/video_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Store uploaded video temporarily
uploaded_video = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global uploaded_video
    video_file = request.files['video']
    
    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        video_file.save(temp_file.name)
        temp_path = temp_file.name
        uploaded_video = temp_path  # Store the path

    try:
        # Get video duration and create a preview
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Create a preview (first frame)
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Could not read video frame'}), 400

        # Resize frame for preview if too large
        max_dimension = 800
        height, width = frame.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        preview_b64 = base64.b64encode(buffer).decode('utf-8')

        # Read the video file for response
        with open(temp_path, 'rb') as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode('utf-8')

        return jsonify({
            'video': video_b64,
            'preview': preview_b64,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames
        })

    finally:
        cap.release()

@app.route('/process', methods=['POST'])
def process():
    global uploaded_video
    if not uploaded_video:
        return jsonify({'error': 'No video uploaded'}), 400

    data = request.get_json()
    start_time = data.get('startTime', 0)
    end_time = data.get('endTime', 0)

    try:
        # Open video file
        cap = cv2.VideoCapture(uploaded_video)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read frames for the selected segment
        frames = []
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            current_frame += 1

        if not frames:
            return jsonify({'error': 'No frames found in selected segment'}), 400

        # Process the frames
        processed_frames = process_video_with_tracknet(frames)

        # Create output video
        output_path = os.path.join(OUTPUT_DIR, f"processed_video_{int(os.path.getmtime(uploaded_video))}.mp4")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=1)
        
        for frame in processed_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
        writer.close()

        # Read the processed video
        with open(output_path, 'rb') as f:
            output_video = f.read()
        video_b64 = base64.b64encode(output_video).decode('utf-8')

        return jsonify({'video': video_b64})

    finally:
        cap.release()

if __name__ == "__main__":
    app.run(debug=True)