import numpy as np
import os
from scipy.misc import imread, imsave
from keras.preprocessing import image
from skimage import exposure

AUGMENTATION_COUNT = 3
DATA_DIR = 'data'
TO_DIR = 'crop'

def get_image(location):
    img = imread(location)
    img = img / 255.
    return img

def random_flip(img, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
    return img

def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    x = x.reshape((h,w,1))
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis,fill_mode, cval)
    return x[:,:,0]

def random_rotate(img, rotate_limit=(-90, 90), u=0.5):
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = rotate(img, theta)
    return img

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    x = x.reshape((h,w,1))
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x[:,:,0]

def random_shift(img, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
    return img

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    x = x.reshape((h,w,1))
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x[:,:,0]

def random_zoom(img, zoom_range=(0.8, 1.0), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
    return img

def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    x = x.reshape((h,w,1))
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x[:,:,0]

def random_shear(img, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
    return img

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        gray = img
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img

def combine(img):
    const_flip = np.random.random()
    const_rotate = np.random.random()
    const_shift = np.random.random()
    const_zoom = np.random.random()
    const_shear = np.random.random()
    const_brightness = np.random.random()
    const_contrast = np.random.random()
    img = random_flip(img,u=const_flip)
    img = random_rotate(img,u=const_rotate)
    img = random_shift(img,u=const_shift)
    img = random_zoom(img,u=const_zoom)
    img = random_shear(img,u=const_shear)
    img = random_brightness(img,u=const_brightness)
    return random_contrast(img,u=const_contrast)

for img in os.listdir(DATA_DIR):
    dir_combine = os.path.join(DATA_DIR,img)
    to_combine = os.path.join(TO_DIR,img)
    fname, ext = os.path.splitext(to_combine)
    real_img = get_image(dir_combine)
    if real_img.shape[0] < 100:
        continue
    imsave('%s%s'%(fname,ext),real_img)
    print(dir_combine,real_img.shape)
    for i in range(AUGMENTATION_COUNT):
        imsave('%s_aug_%d%s'%(fname,i,ext),combine(real_img))
