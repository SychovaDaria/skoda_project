"""
A quick code to test hdr capabilites of OpenCV.

Author: Josef Kahoun
Date 19. 8. 2024
"""

import cv2
import numpy as np

# Load images
img_fn = ['../test_img/hdr_imgs/hdr_1.jpeg','../test_img/hdr_imgs/hdr_2.jpeg', '../test_img/hdr_imgs/hdr_3.jpeg', '../test_img/hdr_imgs/hdr_4.jpeg']
img_list = [cv2.imread(fn) for fn in img_fn]
exp_times = np.array([1/15, 1/5, 1/1500, 1/8000], dtype = np.float32)

# Merge images
# Debevec's algorithm
debevec = cv2.createMergeDebevec()
hdr_debevec = debevec.process(img_list, times=exp_times.copy())
# Tonemap HDR image
tonemap1 = cv2.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())

# Mertenes's algorithm
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Save images
# WARNING: the imgs are blurred due to being taken by hand
cv2.imwrite('../test_img/hdr_imgs/hdr_debevec.jpg', res_debevec * 255)
cv2.imwrite('../test_img/hdr_imgs/hdr_mertens.jpg', res_mertens * 255)