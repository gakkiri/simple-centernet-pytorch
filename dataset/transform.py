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


