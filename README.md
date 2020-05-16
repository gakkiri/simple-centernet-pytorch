# simple-centernet-pytorch
>> This is just a simple implementation of "CenterNet:Objects as Points".  
>> office: https://github.com/xingyizhou/CenterNet


# Motivation

Unlike the other repo, I just used pure pytorch. Unlike most implementation base on detectron2 or mmdetection, I made the code as simple as possible to understand, rarely trick, as close to the original performance as possible(I hope so).  
Anyway, if someone looks at the code and sees any bugs, feel free to let me know.  

# Todo

- [x] Be able to train smoothly and converge.
- [x] Inference.
- [ ] COCO and PASCAL_VOC evaluate.
- [ ] Improve performance
- [ ] So on.

# Requirements

Win10 or Linux.
```
torch >= 1.2.0  
torchvision>=0.4.1  
timm >= 0.1.14
Pillow >= 6.2.2
opencv-python >= 4.2.0
albumentations >= 0.4.5
```

# How to train

Modify ```config/{}.py``` according to your needs, then ```python train_{}.py```.

# How to inference

Modify this script ```inference.py``` that is easy to understand, then ```python inference.py```.

# Some simple results


