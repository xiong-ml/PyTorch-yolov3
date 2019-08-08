import argparse
import sys
import os
import numpy as np

CURRENT_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(CURRENT_DIR)

from video_detect.process_video import process_video_subdirs, get_image
from video_detect.detect_image import DetectModel

from utils.utils import load_classes

# 直接调用 python video_detect/detect_video.py --video_folder ../Mask_RCNN/samples/helmet/videos/ --output_folder /home/share
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--output_folder", type=str, default="output/video", help="output path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    detect_model = DetectModel(
        model_def=opt.model_def, weights_path=opt.weights_path, 
        classes=classes, conf_thres=opt.conf_thres, 
        nms_thres=opt.nms_thres, img_size=opt.img_size
    )

    # blue yellow  white red none
    colors = [(0, 0, 255), (255, 211, 0), (210, 210, 210), (255, 0, 0), (87, 112, 255)]

    def handle_images(images):
        image_infos = detect_model.detect(images)

        target_images = []
        for i, image_info in enumerate(image_infos):
            res_image = get_image(
                images[i], "", image_info["rois"], None, image_info["class_ids"], image_info["scores"], classes, 
                colors=colors, 
                text_colors=colors,
                show_score=False
            )
            target_images.append(np.asarray(res_image) if res_image is not None else images[i])
            
        return target_images

    process_video_subdirs(opt.video_folder, opt.output_folder, 2, handle_images)