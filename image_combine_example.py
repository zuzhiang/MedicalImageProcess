import numpy as np
import skimage.util as su
import matplotlib.pyplot as plt


def paste(img, canvas, i, j, method='replace', export_dtype='float'):
    """ paste `img` on `canvas` with its left-top corner at (i, j) """
    # check dtypes
    img = su.img_as_float(img)
    canvas = su.img_as_float(canvas)
    # check shapes
    if len(img.shape) != 2 or len(img.shape) != 3:
        if len(canvas.shape) != 2 or len(canvas.shape) != 3:
            assert AttributeError('dimensions of input images not all equal to 2 or 3!')
    # check channels
    # all grayscale image
    if len(img.shape) == 2 and len(canvas.shape) == 2:
        pass
    # `img` color image, possible with alpha channel; `canvas` grayscale image
    elif len(img.shape) == 3 and len(canvas.shape) == 2:
        c = img.shape[-1]
        if c == 3:
            canvas = np.stack([canvas]*c, axis=-1)
        if c == 4:
            canvas = np.stack([canvas]*(c-1)+[np.ones((canvas.shape[0], canvas.shape[1]))], axis=-1)
    # `canvas` color image, possible with alpha channel; `img` grayscale image
    elif len(img.shape) == 2 and len(canvas.shape) == 3:
        c = canvas.shape[-1]
        if c == 3:
            img = np.stack([img]*c, axis=-1)
        if c == 4:
            img = np.stack([img]*(c-1)+[np.ones((img.shape[0], img.shape[1]))], axis=-1)
    # all color image
    elif len(img.shape) == 3 and len(canvas.shape) == 3:
        if img.shape[-1] == 3 and canvas.shape[-1] == 4:
            img = np.concatenate([img, np.ones((img.shape[0], img.shape[1], 1))], -1)
        elif img.shape[-1] == 4 and canvas.shape[-1] == 3:
            canvas = np.concatenate([canvas, np.ones((canvas.shape[0], canvas.shape[1], 1))], -1)
        elif img.shape[-1] == canvas.shape[-1]:
            pass
        else:
            assert ValueError('channel number should equal to 3 or 4!')
    # get shapes
    h_i, w_i = img.shape[:2]
    h_c, w_c = canvas.shape[:2]
    # find extent of `img` on `canvas`
    i_min = np.max([0, i])
    i_max = np.min([h_c, i+h_i])
    j_min = np.max([0, j])
    j_max = np.min([w_c, j+w_i])
    # paste `img` on `canvas`
    if method == 'replace':
        canvas[i_min:i_max, j_min:j_max] = img[i_min-i:i_max-i, j_min-j:j_max-j]
    elif method == 'add':
        canvas[i_min:i_max, j_min:j_max] += img[i_min-i:i_max-i, j_min-j:j_max-j]
    else:
        raise ValueError('no such method!')
    # return `canvas`
    if export_dtype == 'float':
        return canvas
    elif export_dtype == 'ubyte':
        return su.img_as_ubyte(canvas)
    else:
        raise ValueError('no such data type for exporting!')


def combine_avg(imgs, num_w=10, strides=(10, 10), padding=5, bg_level_1=1.0, bg_level_2=1.0, export_dtype='float'):
    """ paste contents of `imgs` on a single image with `strides` """
    # dtypes check
    imgs = [su.img_as_float(img) for img in imgs]
    # shapes check
    shapes = [img.shape for img in imgs]
    if not all([len(s) == 2 or len(s) == 3 for s in shapes]):
        assert AttributeError('dimensions of imgs not all 2 or 3!')
    # find the shape of canvas
    n = len(imgs)
    num_h = (n - 1) // num_w + 1
    h = strides[0]*(num_h-1) + shapes[-1][0]
    w = strides[1]*(num_w-1) + shapes[-1][1]
    lt_poses = [(strides[0]*i, strides[1]*j) for i in range(num_h) for j in range(num_w) if i*num_w+j<n]
    # canvas initialization
    if all([len(s)==2 for s in shapes]):
        canvas = np.zeros((h, w), np.float)
        fig = np.zeros((h+2*padding, w+2*padding), np.float) + bg_level_2
    else:
        c = max([s[-1] for s in shapes if len(s)==3])
        canvas = np.zeros((h, w, c), np.float)
        fig = np.zeros((h+2*padding, w+2*padding, c), np.float) + bg_level_2
    # paste `imgs` on `canvas`, average overlapping areas (using ones matrix `counter` to record overlapping times)
    counter = np.zeros((h, w))
    for i, lt_pos in enumerate(lt_poses):
        canvas = paste(imgs[i], canvas, lt_pos[0], lt_pos[1], 'add')
        counter = paste(np.ones((imgs[i].shape[0], imgs[i].shape[1])), counter, lt_pos[0], lt_pos[1], 'add')
    canvas[counter<0.5] = bg_level_1
    counter[counter<0.5] = 1.0
    if len(canvas.shape) == 2:
        canvas = canvas / counter
    else:
        canvas = canvas / np.expand_dims(counter, -1)
    # add padding, i.e. paste `canvas` on `fig`
    fig = paste(canvas, fig, padding, padding)
    # return
    if export_dtype == 'float':
        return fig
    elif export_dtype == 'ubyte':
        return su.img_as_ubyte(fig)
    else:
        raise ValueError('no such data type for exporting!')


black = np.zeros((3, 3))
white = np.ones((3, 3))
b = combine_avg([black, white]*15, 5, (3, 3), 0)

plt.imshow(b)
plt.show()
