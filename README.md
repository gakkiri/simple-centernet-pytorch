# simple-centernet-pytorch
> This is just a simple implementation of "CenterNet:Objects as Points".  
> office: https://github.com/xingyizhou/CenterNet


# Motivation

Unlike the other repo, I just used pure pytorch. Unlike most implementation base on detectron2 or mmdetection, I made the code as simple as possible to understand, no trick, as close to the original performance as possible(I hope so).  
Anyway, if someone looks at the code and sees any bugs, feel free to let me know.  

# Todo

- [x] Be able to train smoothly and converge.
- [x] Inference.
- [ ] COCO and PASCAL_VOC evaluate.
- [ ] Improve performance.
- [ ] More backbone.
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

![e1](https://raw.githubusercontent.com/gakkiri/simple-centernet-pytorch/master/asserts/inf.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![e2](https://raw.githubusercontent.com/gakkiri/simple-centernet-pytorch/master/asserts/inf2.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![e3](https://raw.githubusercontent.com/gakkiri/simple-centernet-pytorch/master/asserts/inf3.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
