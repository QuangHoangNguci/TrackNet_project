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

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default to 30fps if invalid
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return jsonify({'error': 'Invalid video file or empty video'}), 400

        # Validate time range
        if end_time <= start_time:
            end_time = total_frames / fps
        if start_time < 0:
            start_time = 0
        if end_time > total_frames / fps:
            end_time = total_frames / fps

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
                
            # Resize frame to 1280x720
            height, width = frame.shape[:2]
            if width != 1280 or height != 720:
                # Calculate aspect ratio
                aspect_ratio = width / height
                if aspect_ratio > 16/9:  # Wider than 16:9
                    new_width = 1280
                    new_height = int(1280 / aspect_ratio)
                else:  # Taller than 16:9
                    new_height = 720
                    new_width = int(720 * aspect_ratio)
                
                # Resize maintaining aspect ratio
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Create black background
                background = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                # Calculate position to paste resized frame
                x_offset = (1280 - new_width) // 2
                y_offset = (720 - new_height) // 2
                
                # Paste resized frame onto background
                background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
                frame = background
            
            frames.append(frame)
            current_frame += 1

        if not frames:
            return jsonify({'error': 'No frames found in selected segment'}), 400

        # Process the frames
        try:
            processed_frames, bounce_infos = process_video_with_tracknet(frames)
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500

        # Create output video
        output_path = os.path.join(OUTPUT_DIR, f"processed_video_{int(os.path.getmtime(uploaded_video))}.mp4")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=1)
        
        for frame in processed_frames:
            if frame is not None and frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
        writer.close()

        # Read the processed video
        with open(output_path, 'rb') as f:
            output_video = f.read()
        video_b64 = base64.b64encode(output_video).decode('utf-8')

        # Extract bounce frames
        bounce_frames = []
        for info in bounce_infos:
            try:
                idx = info['frame_idx']
                if idx < len(processed_frames):
                    frame = processed_frames[idx]
                    if frame is not None and frame.size > 0:
                        x, y = info['pos']
                        h, w = frame.shape[:2]
                        x1 = max(0, x-250)
                        y1 = max(0, y-250)
                        x2 = min(w, x+250)
                        y2 = min(h, y+250)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            _, buffer = cv2.imencode('.jpg', crop)
                            img_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            minimap = info.get('minimap')
                            minimap_zoom_b64 = None
                            if minimap is not None and minimap.size > 0:
                                h_minimap, w_minimap = minimap.shape[:2]
                                mask = cv2.inRange(minimap, (0, 200, 200), (0, 255, 255))
                                ys, xs = np.where(mask > 0)
                                if len(xs) > 0 and len(ys) > 0:
                                    cx = int(np.mean(xs))
                                    cy = int(np.mean(ys))
                                    crop_size = 550
                                    half_crop = crop_size // 2
                                    mx1 = max(0, cx-half_crop)
                                    my1 = max(0, cy-half_crop)
                                    mx2 = min(w_minimap, cx+half_crop)
                                    my2 = min(h_minimap, cy+half_crop)
                                    minimap_crop = minimap[my1:my2, mx1:mx2]
                                    if minimap_crop.size > 0:
                                        minimap_crop = cv2.resize(minimap_crop, (500, 500))
                                        _, buf2 = cv2.imencode('.jpg', minimap_crop)
                                        minimap_zoom_b64 = base64.b64encode(buf2).decode('utf-8')
                            
                            bounce_frames.append({
                                'image': img_b64,
                                'time': idx/fps,
                                'inout': info['inout'],
                                'minimap': minimap_zoom_b64
                            })
            except Exception as e:
                print(f"Error processing bounce frame: {str(e)}")
                continue

        return jsonify({'video': video_b64, 'bounce_frames': bounce_frames})

    except Exception as e:
        print(f"Error in process route: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    app.run(debug=True)