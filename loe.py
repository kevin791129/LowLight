from PIL import Image
import numpy as np

def loe(im_original, im_enhanced):
    r = 50 / min(im_original.size)
    w = round(im_original.size[0] * r)
    h = round(im_original.size[1] * r)
    l = im_original.resize((w, h), Image.LANCZOS)
    l_max = np.max(l, axis=2).flatten()
    le = im_enhanced.resize((w, h), Image.LANCZOS)
    le_max = np.max(le, axis=2).flatten()
    rd = 0
    len = w * h
    for i in range(len):
        x = l_max[i] >= l_max
        y = le_max[i] >= le_max
        rd += (x ^ y).sum()
    return rd / len