import cv2
import os
import numpy as np
import numpy.random as random

INTERVAL = 1

# 其实出现这么多的坑就是因为对opencv读取图片的颜色空间和数据格式不清楚，他们里面有BGR和RGB颜色空间，而一般cv2.imread()读取的图片都是BGR颜色空间的图片，cv2.VideoCapture()获取的视频帧是RGB颜色空间的图片。PIL（Python Image Library）读取的图片是RGB颜色空间的。 
# opencv读取的图片不管是视频帧还是图片都是矩阵形式，即np.array，转PIL.Image格式用PIL.Image.fromarray()函数即可。


# 从video_path 读取一个视频，返回images
def read_images_from_path(video_path):
    print('read images from ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    images = []
    i = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if i % INTERVAL == 0:
            images.append(image)
        i += 1
        
    fps =int(vidcap.get(cv2.CAP_PROP_FPS))
    size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vidcap.release()
    return images, fps // INTERVAL, size

# 将images写入到video_path，生成视频文件
def write_images_into_path(images, fps, size, video_path):
    print('write images into ', video_path)
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(images)):
        # writing to a image array
        out.write(images[i])
    out.release()
    print('write images into ', video_path, 'is complete')
    
VIDEO_EXTENSIONS = ['.mp4']
def process_video_subdirs(root_dir, target_dir, batch_size, handler):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        print('enter dir path', dirpath)
        for filename in filenames:
            print('will process file %s' % filename)
            if os.path.splitext(filename)[1] in VIDEO_EXTENSIONS:
                images, fps, size = read_images_from_path(os.path.join(dirpath, filename))
                print(type(images[0]) if len(images) > 0 else '')
                target_images = []
                for i in range(len(images) // batch_size):
                    target_images.extend(handler(images[i * batch_size: i * batch_size + batch_size]))
                if dirpath.startswith(root_dir):
                    target_filename = os.path.join(target_dir, filename)
                    print('will write processed file to %s' % target_filename)
                    write_images_into_path(target_images, fps, size, target_filename)
 

from PIL import Image, ImageDraw, ImageFont
import colorsys

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def get_image(image, image_name, boxes, masks, class_ids, scores, class_names, 
              colors = None, 
              text_colors = None,
              show_score=True,
              filter_classs_names=None,
              mask_only=False,
              show_mask=True,
              scores_thresh=0.2):
    """
        image: image array
        image_name: image name
        boxes: [num_instance, (x1, y1, x2, y2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        mask_only: save mask with black background

        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
   
    useful_mask_indices = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        useful_mask_indices.append(i)

    print('get useful mask indices:', useful_mask_indices)
    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return

    if not colors:
        colors = random_colors(len(useful_mask_indices))
    
    if not text_colors:
        text_colors = [(255, 255, 255) for i in range(len(class_names))]
        
    if not mask_only:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)

    if show_mask and masks is not None:
        for index, value in enumerate(useful_mask_indices):
            masked_image = apply_mask(masked_image, masks[:, :, value], colors[index])

    masked_image = Image.fromarray(masked_image)

    if mask_only:
        # masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
        return np.asarray(masked_image)

    draw = ImageDraw.Draw(masked_image)
    # colors = np.array(colors).astype(int) * 255

    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        score = scores[value]
        label = class_names[class_id]

        print('get box val', str(boxes[value]))
        x1, y1, x2, y2 = boxes[value]

        # draw rectangle for ROI
        color = tuple(colors[class_id])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        # Label
        # font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
        # font = ImageFont.load_default()
        font = ImageFont.truetype(os.path.join(os.getcwd(), "../Mask_RCNN/fonts/OpenSans-Bold.ttf"), 20, encoding="unic")

        if show_score:
            draw.text((x1, y1), "%s %f" % (label, score), text_colors[class_id], font)
        else:
            draw.text((x1, y1), "%s" % (label), text_colors[class_id], font)

    # masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
    return masked_image

# 对images进行 检测，并返回结果
def detect_images(model, images, class_names, **options):
    # skimages = [img_as_float(image) for image in images]
    results = model.detect(images, verbose=1)
    targets = []
    for i, r in enumerate(results):
        target = get_image(
           images[i], "filename", r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, 
           mode=1, 
           **options
        )
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                   dataset_val.class_names, r['scores'], ax=get_ax())
        
        if target is None:
            print('This image is not valid')
            target = images[i]
        
        # visualize.display_instances(skimages[i], r['rois'], r['masks'], r['class_ids'], 
        #                class_names, r['scores'])
            
        targets.append(target)
    return targets