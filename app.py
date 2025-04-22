import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import subprocess
import os


class VideoProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing Application")
        self.root.geometry("500x250")

        # Nút chọn video
        self.select_button = tk.Button(root, text="Select Video", command=self.select_video, height=2)
        self.select_button.pack(pady=20)

        # Thanh tiến trình
        self.progress_bar = Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=20)

        # Nút bắt đầu xử lý video
        self.process_button = tk.Button(root, text="Process Video", command=self.start_processing_video, height=2,
                                        state=tk.DISABLED)
        self.process_button.pack(pady=20)

        self.video_path = None
        self.output_video_path = None

    def select_video(self):
        """Hàm chọn video"""
        self.video_path = filedialog.askopenfilename(title="Select a video file",
                                                     filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if self.video_path:
            # Hiển thị thông báo khi video được chọn
            messagebox.showinfo("Success", "Video uploaded successfully!")
            # Kích hoạt nút "Process Video"
            self.process_button.config(state=tk.NORMAL)
        else:
            # Nếu không có video được chọn, vô hiệu hóa nút "Process Video"
            self.process_button.config(state=tk.DISABLED)

    def start_processing_video(self):
        """Hàm xử lý video"""
        if self.video_path:
            # Vô hiệu hóa nút "Select Video" và "Process Video" khi đang xử lý
            self.select_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.DISABLED)

            # Xác định đường dẫn video đầu ra
            self.output_video_path = os.path.splitext(self.video_path)[0] + "_processed.avi"

            # Chạy câu lệnh xử lý video
            self.process_video()

            # Cập nhật thanh tiến trình
            for i in range(100):
                self.progress_bar['value'] = i
                self.root.update_idletasks()

            # Sau khi xử lý xong, kích hoạt lại nút "Select Video" và "Process Video"
            self.select_button.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)

            # Hiển thị thông báo khi xử lý xong
            messagebox.showinfo("Processing Complete", "Video processing is complete!")

    def process_video(self):
        """Hàm xử lý video thông qua câu lệnh shell"""
        # Đoạn command thực thi xử lý video
        command = [
            'python', 'main.py',  # Chạy file chính
            '--path_ball_track_model', '/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/model_best.pt',
            '--path_court_model', '/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/model_tennis_court_det.pt',
            '--path_bounce_model', '/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/bounce_detection_weights.cbm',
            '--path_input_video', self.video_path,
            '--path_output_video', self.output_video_path
        ]

        try:
            # In ra câu lệnh để kiểm tra
            print(f"Running command: {' '.join(command)}")

            # Chạy subprocess để thực thi câu lệnh
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Kiểm tra kết quả từ câu lệnh
            if result.returncode == 0:
                print("Video processed successfully!")
                print(f"Output: {result.stdout}")
            else:
                print(f"Error: {result.stderr}")
                messagebox.showerror("Error", f"Processing failed: {result.stderr}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessingApp(root)
    root.mainloop()
