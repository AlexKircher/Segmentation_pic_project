from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torchvision.transforms as T
from torchvision import models


def decode_segmap(image, nc=21):
    label_colors = np.array([(200,200,200),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    a = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        if l==0:
            a[idx]=0
        else:
            a[idx]=255


    rgb = np.stack([r, g, b, a], axis=2)
    return rgb

fcn= models.segmentation.deeplabv3_resnet101(pretrained=True).eval() #добавляем предобученную модель

#загрузка изображений

img = Image.open("./img/road_with_car.png")
img = img.convert("RGB")
transform = T.Compose([T.Resize(256),
                 #T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp= transform(img).unsqueeze(0)
out = fcn(inp)['out']
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
img_mask = decode_segmap(om)
print(img_mask.shape)
img_mask = Image.fromarray(img_mask,"RGBA")
img = img.convert("RGBA")
img_mask.save("img.png","PNG")
img = img.resize(img_mask.size)
print(img.size)
print(img_mask.size)

img_f  =Image.blend(img, img_mask,0.4)
plt.imshow(img_f)

plt.show()