import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import threading

# Thêm toàn bộ pipeline import như trước đó
import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from main import read_video, main, write
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect
import torch

# Các hàm như read_video(), get_court_img(), main(), write() giữ nguyên

def process_video(path_input_video, path_output_video, path_ball_model, path_court_model, path_bounce_model, update_progress=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = read_video(path_input_video)
    scenes = scene_detect(path_input_video)

    if update_progress: update_progress('Ball Detection...')
    ball_detector = BallDetector(path_ball_model, device)
    ball_track = ball_detector.infer_model(frames)

    if update_progress: update_progress('Court Detection...')
    court_detector = CourtDetectorNet(path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    if update_progress: update_progress('Player Detection...')
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    if update_progress: update_progress('Bounce Detection...')
    bounce_detector = BounceDetector(path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    if update_progress: update_progress('Rendering video...')
    imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                    draw_trace=True)

    write(imgs_res, fps, path_output_video)
    if update_progress: update_progress('Done!')


def run_gui():
    def select_file(var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def run_processing():
        input_video = path_input.get()
        output_video = path_output.get()
        ball_model = path_ball.get()
        court_model = path_court.get()
        bounce_model = path_bounce.get()

        if not all([input_video, output_video, ball_model, court_model, bounce_model]):
            messagebox.showerror("Error", "Please fill in all fields.")
            return

        def update_status(msg):
            progress_var.set(msg)
            root.update_idletasks()

        def thread_target():
            try:
                process_video(input_video, output_video, ball_model, court_model, bounce_model, update_status)
                messagebox.showinfo("Success", "Video processing completed.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=thread_target).start()

    root = tk.Tk()
    root.title("Tennis Video Processor")
    root.geometry("600x400")

    path_input = tk.StringVar()
    path_output = tk.StringVar()
    path_ball = tk.StringVar()
    path_court = tk.StringVar()
    path_bounce = tk.StringVar()
    progress_var = tk.StringVar()

    entries = [
        ("Input Video", path_input),
        ("Output Video", path_output),
        ("Ball Model Path", path_ball),
        ("Court Model Path", path_court),
        ("Bounce Model Path", path_bounce),
    ]

    for idx, (label_text, var) in enumerate(entries):
        label = tk.Label(root, text=label_text)
        label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')
        entry = tk.Entry(root, textvariable=var, width=50)
        entry.grid(row=idx, column=1, padx=10)
        button = tk.Button(root, text="Browse", command=lambda v=var: select_file(v))
        button.grid(row=idx, column=2)

    tk.Button(root, text="Process Video", command=run_processing, bg="green", fg="white").grid(row=len(entries), column=1, pady=20)
    tk.Label(root, textvariable=progress_var, fg="blue").grid(row=len(entries)+1, column=1)

    root.mainloop()

if __name__ == '__main__':
    run_gui()