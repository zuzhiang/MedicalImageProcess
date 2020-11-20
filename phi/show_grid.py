import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np


def grid2contour(grid, title):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    assert grid.ndim == 3
    x = np.arange(-1, 1, 2.0 / grid.shape[1])
    y = np.arange(-1, 1, 2.0 / grid.shape[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2

    plt.figure()
    plt.contour(X, Y, Z1, 15, levels=50, colors='k') #改变levels的值，可以改变形变场的外貌
    plt.contour(X, Y, Z2, 15, levels=50, colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title(title)
    plt.show()


def show_grid():
    img = sitk.ReadImage(r"C:\Users\zuzhiang\Desktop\7_flow.nii.gz")
    img_arr = sitk.GetArrayFromImage(img)[:,:,0,:2]
    img_shape = img_arr.shape
    print("shape: ", img_shape)

    # 起点、终点、步长（可为小数）
    x = np.arange(-1, 1, 2 / img_shape[1])
    y = np.arange(-1, 1, 2 / img_shape[0])
    X, Y = np.meshgrid(x, y)
    regular_grid = np.stack((X, Y), axis=2)
    grid2contour(regular_grid, "regular_grid")

    rand_field = np.random.rand(*img_shape[:2], 2)  # 参数前加*是以元组形式导入
    rand_field_norm = rand_field.copy()
    rand_field_norm[:, :, 0] = rand_field_norm[:, :, 0] * 2 / img_shape[1]
    rand_field_norm[:, :, 1] = rand_field_norm[:, :, 1] * 2 / img_shape[0]
    sampling_grid = regular_grid + rand_field_norm
    grid2contour(sampling_grid, "sampling_grid")

    img_arr[..., 0] = img_arr[..., 0] * 2 / img_shape[1]
    img_arr[..., 1] = img_arr[..., 1] * 2 / img_shape[0]
    img_grid = regular_grid + img_arr
    grid2contour(img_grid, "img_grid")


if __name__ == "__main__":
    show_grid()
    print("end")
