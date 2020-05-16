import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from dataset.utils import gaussian_radius, draw_umich_gaussian
import os
import cv2
import xml.etree.ElementTree as ET
from dataset.transform import Transform

opj = os.path.join


class VOCDataset(Dataset):
    CLASSES_NAME = (
        '__back_ground__', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='../dataset/voc/VOC2012', resize_size=(800, 1024), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835)):
        self.root = root
        self.image_folder = opj(self.root, 'JPEGImages')
        self.annotation = opj(self.root, 'Annotations')
        self.mode = mode

        self.transform = Transform('pascal_voc')
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

        with open(opj(self.root, 'ImageSets', 'Main', f'{self.mode}.txt')) as f:
            self.samples = f.read().split()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(opj(self.image_folder, f'{sample}.jpg')).convert('RGB')
        label = self.parse_annotation(opj(self.annotation, f'{sample}.xml'))
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}
        if self.mode == 'train':
            img, boxes = self.transform(img, label['boxes'], label['labels'])
        else:
            boxes = np.array(label['boxes'])
        boxes_w, boxes_h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
        ct = np.array([(boxes[..., 0] + boxes[..., 2]) / 2,
                       (boxes[..., 1] + boxes[..., 3]) / 2], dtype=np.float32).T

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = label['labels']

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((20, output_h, output_w), dtype=np.float32)
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if hm[cls_id - 1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return img, boxes, classes, hm, info

    def preprocess_img_boxes(self, image, input_ksize, boxes=None):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        _pad = 32  # 32

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = _pad - nw % _pad
        pad_h = _pad - nh % _pad

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = list()
        labels = list()
        difficulties = list()
        for object in root.iter('object'):

            difficult = int(object.find('difficult').text == '1')

            label = object.find('name').text.lower().strip()
            if label not in self.CLASSES_NAME:
                continue

            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.category2id[label])
            difficulties.append(difficult)

        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // 4 - hm.shape[2]), 0, int(max_h // 4 - hm.shape[1])),
                                        value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)

        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos


if __name__ == '__main__':
    ds = VOCDataset('../../../PASCAL_VOC/VOCdevkit/VOC2012', resize_size=(512, 512))
    dl = DataLoader(ds, batch_size=12, collate_fn=ds.collate_fn)
    batch = next(iter(dl))

    # print(len(dl))
    # import tqdm
    # for i, data in enumerate(tqdm.tqdm(dl)):
    #     # pass
    #     imgs, boxes, classes, hms, infos = data
    #     for b in range(hms.size(0)):
    #         if hms[b].eq(1).sum(0).gt(0).sum() != classes[b].gt(0).sum():
    #             import pdb; pdb.set_trace()
