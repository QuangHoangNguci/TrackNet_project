import cv2
import numpy as np
import torch
from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect, get_court_img


def process_video_with_tracknet(frames):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Scene detection
    scenes = scene_detect(frames)

    # Ball detection
    ball_detector = BallDetector('/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/model_best.pt', device)
    ball_track = ball_detector.infer_model(frames)

    # Court detection
    court_detector = CourtDetectorNet('/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/model_tennis_court_det.pt', device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    # Person detection
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    # Bounce detection
    bounce_detector = BounceDetector('/Users/QuangHoang/PycharmProjects/pythonProject/TrackNet_project/model_weights/bounce_detection_weights.cbm')
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # Process frames with overlays
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices]

    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]
        eps = 1e-15
        scene_rate = sum_track / (len_track + eps)
        if scene_rate > 0.5:
            court_img = get_court_img()
            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i].copy()
                inv_mat = homography_matrices[i]
                if ball_track[i][0]:
                    for j in range(0, 7):
                        if i - j >= 0 and ball_track[i - j][0]:
                            draw_x = int(ball_track[i - j][0])
                            draw_y = int(ball_track[i - j][1])
                            img_res = cv2.circle(img_res, (draw_x, draw_y),
                                                 radius=3, color=(0, 255, 0), thickness=2)
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                             radius=0, color=(0, 0, 255), thickness=10)
                height, width, _ = img_res.shape
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                           radius=0, color=(0, 255, 255), thickness=50)
                minimap = court_img.copy()
                persons = persons_top[i] + persons_bottom[i]
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                             radius=0, color=(255, 0, 0), thickness=80)
                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)
        else:
            imgs_res.extend(frames[scenes[num_scene][0]:scenes[num_scene][1]])
    return imgs_res