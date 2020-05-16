from torchvision.datasets import CocoDetection
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from dataset.utils import gaussian_radius, draw_umich_gaussian
from dataset.transform import Transform




class COCODataset(CocoDetection):
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, resize_size=(800, 1024), mode='TRAIN',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835)):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.transform = Transform(mode)
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4
        self.mode = mode.lower()

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)
        # print(ann)
        info = {'index': index}

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        boxes_w, boxes_h = boxes[..., 2], boxes[..., 3]

        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        # x, y
        ct = np.array([(boxes[..., 0] + boxes[..., 2]) / 2,
                       (boxes[..., 1] + boxes[..., 3]) / 2], dtype=np.float32).T
        img = np.array(img)
        h, w, _ = img.shape
        info['raw_height'], info['raw_width'] = h, w
        if self.mode == 'TRAIN':
            img, boxes = self.transform(img, boxes)

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        # img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((80, output_h, output_w), dtype=np.float32)
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
            if hm[cls_id-1, ct_int[1], ct_int[0]] != 1:
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

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False
        if self._has_only_empty_bbox(annot):
            return False
        return True

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
                torch.nn.functional.pad(hm, (0, int(max_w//4 - hm.shape[2]), 0, int(max_h//4 - hm.shape[1])), value=0.))

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


if __name__ == "__main__":
    import tqdm
    dataset = COCODataset("F:/COCO/train2017", "F:/COCO/annotations/instances_train2017.json",
                          resize_size=[512, 512])
    dl = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    dataset.__getitem__(37)
    # for i, data in enumerate(tqdm.tqdm(dl)):
    #     # pass
    #     imgs, boxes, classes, hms, infos = data
    #     for b in range(hms.size(0)):
    #         if hms[b].eq(1).sum(0).gt(0).sum() != classes[b].gt(0).sum():
    #             import pdb; pdb.set_trace()

