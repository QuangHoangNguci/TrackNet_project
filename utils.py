import cv2
import numpy as np
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from court_reference import CourtReference

'''
def scene_detect(path_video):

    #Split video to disjoint fragments based on color histograms

    video_manager = VideoManager([path_video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    scenes = [[x[0].frame_num, x[1].frame_num]for x in scene_list]
    return scenes
'''
def scene_detect(frames):
    # Instead of using scenedetect, treat the entire video as one scene
    # Return a list of tuples [(start_frame, end_frame)]
    # For simplicity, assume the whole video is one scene
    return [(0, len(frames))]

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img