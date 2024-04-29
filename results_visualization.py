import os 
import cv2
from scripts.data_converter.visual_utils import *

def visual_results(data_root, result_dir, filename):
    if not os.path.exists(data_root):
        raise ValueError("data_root Not Found!") 
    
    image_path = os.path.join(data_root, "training/image_2", f"{filename}.jpg")
    calib_path = os.path.join(data_root, "training/calib", f"{filename}.txt")
    label_path = os.path.join(result_dir, "data", f"{filename}.txt")

    image = cv2.imread(image_path)
    _, P2, denorm = load_calib(calib_path)
    image = draw_3d_box_on_image(image, label_path, P2, denorm)

    detection_path = os.path.join(result_dir, "detections")
    if not os.path.exists(detection_path):
        os.mkdir(detection_path)
    cv2.imwrite(os.path.join(detection_path, f"{filename}_detection.jpg"), image)

data_root = 'data/dair-v2x-i-kitti'
result_dir = 'outputs'
filename = '000018'
if __name__ == "__main__":
    visual_results(data_root, result_dir, filename)