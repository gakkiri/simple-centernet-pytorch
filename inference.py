from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from model.centernet import CenterNet
import matplotlib.pyplot as plt
from pprint import pprint


def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    img_resized = np.array(img.resize((nw, nh)))

    pad_w = _pad - nw % _pad
    pad_h = _pad - nh % _pad

    img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded[:nh, :nw, :] = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}


def show_img(img, boxes, clses, scores):
    boxes, scores = [i.cpu() for i in [boxes, scores]]

    boxes = boxes.long()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(xy=box.tolist(), outline='red', width=3)

    boxes = boxes.tolist()
    scores = scores.tolist()
    plt.figure(figsize=(10, 10))
    for i in range(len(boxes)):
        plt.text(x=boxes[i][0], y=boxes[i][1], s='{}: {:.4f}'.format(clses[i], scores[i]), wrap=True, size=15,
                 bbox=dict(facecolor="r", alpha=0.7))
    plt.imshow(img)
    plt.show()


ckp = torch.load('./ckp/best_checkpoint.pth')
cfg = ckp['config']
pprint(cfg)

img = Image.open('./test/test4.jpg').convert('RGB')
print('preprocessing input...')
img_paded, info = preprocess_img(img, cfg.resize_size)

imgs = [img]
infos = [info]

input = transforms.ToTensor()(img_paded)
input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
inputs = input.unsqueeze(0).cuda()
print('preprocess done!\ninit model...')

model = CenterNet(cfg).cuda()
model.load_state_dict(ckp['model'])
model = model.eval()
print('model done!\ndetecting...')

detects = model.inference(inputs, infos, topK=40, return_hm=False, th=0.25)

for img_idx in range(len(detects)):  # 1
    boxes = detects[img_idx][0]
    scores = detects[img_idx][1]
    clses = detects[img_idx][2]

    img = imgs[img_idx]
    show_img(img, boxes, clses, scores)
