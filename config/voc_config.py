class Config(object):
    CLASSES_NAME = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    # backbone
    slug = 'r50'
    fpn = False
    freeze_bn = True

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
    num_classes = 20
    batch_size = 16  # 128
    root = '../VOCdevkit/VOC2012'
    split = 'trainval'
    resize_size = [512, 512]
    num_workers = 4
    mean = [0.40789654, 0.44719302, 0.47026115]
    std = [0.28863828, 0.27408164, 0.27809835]

    # train
    optimizer = 'AdamW'
    lr = 1e-2
    AMSGRAD = True

    max_iter = 45000
    lr_schedule = 'WarmupMultiStepLR'
    gamma = 0.1
    steps = (35000, 40000)
    warmup_iters = 1000

    apex = False

    # other
    gpu = True
    eval = False
    resume = False
    score_th = 0.1
    down_stride = 4
    topK = 100
    log_dir = './log'
    checkpoint_dir = './ckp'
    log_interval = 20

    def __str__(self):
        s = '\n'
        for k, v in Config.__dict__.items():
            if k[:2] == '__' and k[-2:] == '__':
                continue
            s += k + ':  ' + str(v) + '\n'
        return s
