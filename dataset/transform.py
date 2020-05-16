from albumentations import Compose, BboxParams, \
    RandomBrightnessContrast, GaussNoise, RGBShift, CLAHE, RandomGamma, HorizontalFlip, RandomResizedCrop


class Transform(object):
    def __init__(self, box_format='coco'):
        self.tsfm = Compose([
            HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),
            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    import numpy as np
    import xml.etree.ElementTree as ET

    CLASSES_NAME = (
        '__back_ground__', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def parse_annotation(annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = list()
        labels = list()
        difficulties = list()
        for object in root.iter('object'):

            difficult = int(object.find('difficult').text == '1')

            label = object.find('name').text.lower().strip()
            if label not in CLASSES_NAME:
                continue

            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            difficulties.append(difficult)

        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


    def show_img(img, boxes, clses):
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(xy=box.tolist(), outline='red')
            draw.rectangle(xy=(box + 1).tolist(), outline='red')
            draw.rectangle(xy=(box + 2).tolist(), outline='red')


    o_img = Image.open('F:/torch学习/PASCAL_VOC/VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg').convert('RGB')
    anno = parse_annotation('F:/torch学习/PASCAL_VOC/VOCdevkit/VOC2012/Annotations/2007_000129.xml')

    img = np.array(o_img)
    boxes = anno['boxes']
    labels = anno['labels']

    tsfm = Transform('pascal_voc')
    img, boxes = tsfm(img, boxes, labels)

    draw_img = Image.fromarray(img)
    show_img(draw_img, np.array(boxes), labels)
    draw_img.show()
