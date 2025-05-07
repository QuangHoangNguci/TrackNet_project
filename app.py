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

input_frames = []
fps = 0
video_duration = 0

# Directory to save processed videos
OUTPUT_DIR = "/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/video_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global input_frames, fps, video_duration
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
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

    return jsonify({
        'video': video_b64,
        'duration': video_duration
    })

@app.route('/process', methods=['POST'])
def process():
    global input_frames, fps
    if not input_frames:
        return jsonify({'error': 'No video uploaded'}), 400

    # Get start and end times from request
    data = request.get_json()
    start_time = data.get('startTime', 0)
    end_time = data.get('endTime', video_duration)

    # Convert times to frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Ensure valid frame range
    start_frame = max(0, min(start_frame, len(input_frames)))
    end_frame = max(start_frame, min(end_frame, len(input_frames)))

    # Cut the video frames
    cut_frames = input_frames[start_frame:end_frame]

    # Process the cut video using TrackNet
    processed_frames = process_video_with_tracknet(cut_frames)

    # Convert processed frames to video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_path = temp_file.name
        # Use imageio to write video with H.264 codec
        writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', macro_block_size=1)
        for frame in processed_frames:
            # Convert BGR (OpenCV) to RGB
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