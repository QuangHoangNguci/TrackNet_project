# processor.py
import torch
from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from bounce_detector import BounceDetector
from main import read_video, main, write
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect

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
    imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, draw_trace=True)

    write(imgs_res, fps, path_output_video)
    if update_progress: update_progress('Done!')
