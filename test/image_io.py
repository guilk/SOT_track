import skimage.io
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize

def load_image(filename, color = True):
    '''
    
    :param filename: string 
    :param color: boolean flag for color format. True (default) loads as RGB while False loads as intensity
                    (if image is already grayscale)
    :return: 
            image: an image with type np.float32 in range[0,1]
            of size(H x W x 3) in RGB or 
            of size(H x W x 1) in grayscale
    '''
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def resize_image(im, new_dims, interp_order = 1):
    '''
    Resize an image array with interpolation
    :param im: (H x W x K) ndarray
    :param new_dims: (height, width) tuple of new dimensions
    :param interp_order: interpolation order, default is linear
    :return: 
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    '''
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)
