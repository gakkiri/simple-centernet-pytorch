class Config(object):
    CLASSES_NAME = (
        'person', 'bicycle', 'car', 'motorcycle',
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

    # backbone
    slug = 'r50'
    fpn = False
    freeze_bn = False

    # decoder
    bn_momentum = 0.1

    # head
    head_channel = 64

    # loss
    regr_loss = 'iou'
    loss_alpha = 1.
    loss_beta = 0.1
    loss_gamma = 1.

    # dataset
    num_classes = 80
    batch_size = 64  # 128
    train_imgs_path = '/public/home/jd_ftm/datasets/coco/train2017'
    train_anno_path = '/public/home/jd_ftm/datasets/coco/annotations/instances_train2017.json'
    eval_imgs_path = '/public/home/jd_ftm/datasets/coco/val2017'
    eval_anno_path = '/public/home/jd_ftm/datasets/coco/annotations/instances_val2017.json'
    resize_size = [512, 512]
    num_workers = 4
    mean = [0.40789654, 0.44719302, 0.47026115]
    std = [0.28863828, 0.27408164, 0.27809835]

    # train
    optimizer = 'AdamW'
    lr = 1e-2
    AMSGRAD = True

    gap = max(128 // batch_size, 1)
    max_iter = 126000 * gap
    lr_schedule = 'WarmupMultiStepLR'
    gamma = 0.1
    steps = (81000 * gap, 108000 * gap)
    warmup_iters = 1000 * gap

    apex = False

    # other
    gpu = True
    eval = True
    resume = False
    score_th = 0.1
    down_stride = 4
    topK = 100
    log_dir = './log'
    checkpoint_dir = './ckp'
    log_interval = 100

    def __str__(self):
        s = '\n'
        for k, v in Config.__dict__.items():
            if k[:2] == '__' and k[-2:] == '__':
                continue
            s += k + ':  ' + str(v) + '\n'
        return s

    @staticmethod
    def __todict__():
        _dict = {}
        for k, v in Config.__dict__.items():
            if k[:2] == '__' and k[-2:] == '__':
                continue
            _dict[k] = v
        return _dict
