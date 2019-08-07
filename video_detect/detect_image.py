import os
import torch
import sys
import argparse
import glob

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cur_dir, "../")))
sys.path.append(os.path.abspath(cur_dir))

from models import *
from utils.utils import *
from utils.datasets import *
import datetime
import time

import torch.nn.functional as F
import torchvision.transforms as transforms
from process_video import get_image

class DetectModel:
    def __init__(self, **opt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)

        if opt["weights_path"].endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt["weights_path"])
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt["weights_path"]))

        model.eval()  # Set in evaluation mode
        self.model = model
        self.opt = opt
    
    def detect(self, imgs):
        return detect_images(self.model, imgs, self.opt["conf_thres"], self.opt["nms_thres"], self.opt["img_size"], self.opt["classes"])
    
def resize(image, size=416):
    return F.interpolate(image.unsqueeze(0), size=size).squeeze(0)

def transform_image(image, img_size):
    image = transforms.ToTensor()(image)
    image, _ = pad_to_square(image, 0)
    return resize(image, img_size)

def detect_images(model, imgs, conf_thres, nms_thres, img_size, classes):
    img_tensors = [transform_image(img, img_size) for img in imgs]

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = torch.stack(img_tensors).type(Tensor)

    prev_time = time.time()

    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch, Inference Time: %s" % inference_time)

    image_infos = []
    for img_i, detection in enumerate(detections):
        print("will rescale boxes:", img_size, imgs[img_i].shape[:2])
        detection = rescale_boxes(detection, img_size, imgs[img_i].shape[:2])

        rois = []
        class_ids = []
        class_scores = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            rois.append([int(x1), int(y1), int(x2), int(y2)])
            class_ids.append(int(cls_pred))
            class_scores.append(cls_conf.item())
        image_infos.append({ "rois": np.array(rois), "class_ids": np.array(class_ids), "scores": np.array(class_scores) })
    
    # blue yellow  white red none
    colors = [(0, 196, 214), (255, 119, 0), (210, 210, 210), (255, 0, 0), (87, 112, 255)]

    for i, image_info in enumerate(image_infos):
        res_image = get_image(imgs[i], "", image_info["rois"], None, image_info["class_ids"], image_info["scores"], classes, colors=colors, text_colors=colors)
        res_image.save(os.path.join('output', '%s.jpg' % (str(i))))

# 直接调用 python video_detect/detect_image.py --image_folder ../Mask_RCNN/samples/helmet/samples/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
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

    files = sorted(glob.glob("%s/*.*" % opt.image_folder))
    imgs = [np.asarray(Image.open(file)) for file in files[0: 2]]
    
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    detect_model = DetectModel(
        model_def=opt.model_def, weights_path=opt.weights_path, 
        classes=classes, conf_thres=opt.conf_thres, 
        nms_thres=opt.nms_thres, img_size=opt.img_size
    )
    detect_model.detect(imgs)