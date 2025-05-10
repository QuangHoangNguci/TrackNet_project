import cv2
import numpy as np
import torch
from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect, get_court_img


def is_point_in_court(point, court_polygon):
    # point: (x, y), court_polygon: np.array shape (N, 2)
    # return True if point in polygon
    return cv2.pointPolygonTest(court_polygon, (float(point[0]), float(point[1])), False) >= 0


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

    # Court reference polygon (chuẩn)
    court_ref = CourtReference()
    court_polygon = np.array(court_ref.border_points, dtype=np.float32)  # shape (4,2)

    # Process frames with overlays
    imgs_res = []
    bounce_infos = []  # Lưu thông tin bounce: frame_idx, inout, vị trí, minimap
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
                minimap = court_img.copy()  # minimap 2D cho frame này
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
                    ball_point_arr = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    # Chuyển sang hệ quy chiếu sân chuẩn
                    ball_on_court = cv2.perspectiveTransform(ball_point_arr, inv_mat)[0, 0]
                    # Kiểm tra in/out
                    inout = 'IN' if is_point_in_court(ball_on_court, court_polygon) else 'OUT'
                    # Vẽ chữ lên frame
                    pos = (int(ball_track[i][0]), int(ball_track[i][1]))
                    color = (0, 255, 0) if inout == 'IN' else (0, 0, 255)
                    cv2.putText(img_res, inout, (pos[0]+10, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    # Vẽ lên minimap
                    minimap = cv2.circle(minimap, (int(ball_on_court[0]), int(ball_on_court[1])),
                                           radius=0, color=(0, 255, 255), thickness=50)
                    # Lưu thông tin bounce kèm minimap
                    bounce_infos.append({'frame_idx': i, 'inout': inout, 'pos': pos, 'minimap': minimap.copy()})
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
                minimap_resized = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap_resized
                imgs_res.append(img_res)
        else:
            imgs_res.extend(frames[scenes[num_scene][0]:scenes[num_scene][1]])
    return imgs_res, bounce_infos