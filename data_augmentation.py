import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = -100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


# 数据增强函数，可以对2D/3D图像随机进行直方图均衡化、翻转、平移、放缩、旋转、灰度值归一化等操作
def data_augmentation(path=None, image=None, do_bspline=True, do_flip=False, do_trslt=True, do_scale=True, do_rota=True,
                      show_help_info=False):
    '''
    :param path: 图像路径
    :param image: 输入图像，当path为None使用
    :param do_bspline: 是否做B样条转换，只在为3D图像时使用，因2D时有问题（没效果）
    :param do_flip: 是否做翻转操作
    :param do_trslt: 是否做平移操作
    :param do_scale: 是否做缩放操作
    :param do_rota: 是否做旋转操作
    :param show_help_info: 是否显示帮助信息
    :return:
    '''
    if path is not None:
        image = sitk.ReadImage(path)
    image_arr = sitk.GetArrayFromImage(image)

    # 去除灰度直方图中最亮的0.2%的灰度值
    value, _, count = np.unique(image_arr, return_inverse=True, return_counts=True)
    # axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把numpy数组当成一个一维数组。
    # 累加即当前元素值，等于之前所有元素值之和+当前元素值
    quantiles = np.cumsum(count).astype(np.float32)
    quantiles /= quantiles[-1]
    # 找到灰度直方图中小于0.998的最大位置，并将大于其对应值的值设为该值
    pos = np.where(quantiles < 0.998)[0]
    val_max = value[np.max(pos)]
    image_arr[image_arr > val_max] = val_max

    # 计算窗宽和窗位
    min_val, max_val = np.min(image_arr), np.max(image_arr)
    window_width, window_center = max_val - min_val, (min_val + max_val) / 2

    dim = image.GetDimension()
    if show_help_info:
        print("dim: ", dim)
    transform = sitk.AffineTransform(dim)
    matrix = np.array(transform.GetMatrix()).reshape((dim, dim))
    if dim == 2:
        if do_flip:
            # 翻转（flip）
            if np.random.rand() > 0.5:
                image = image[::-1]  # 沿x轴反转
                if show_help_info:
                    print("x axis flip.")
            if np.random.rand() > 0.5:
                image = image[:, ::-1]  # 沿y轴反转
                if show_help_info:
                    print("y axis flip.")
        if do_trslt:
            # 平移（translation）
            x_trslt, y_trslt = np.random.randint(-20, 20, 2)  # 每个轴的平移范围在[-20,20]像素之间
            transform.SetTranslation((float(x_trslt), float(y_trslt)))
            if show_help_info:
                print("x_trslt: ", x_trslt, "  y_trslt: ", y_trslt)
        if do_scale:
            # 缩放（scale）
            x_scale = 1.0 + random.uniform(-0.1, 0.1)  # 缩放范围为原来的[0.9,1.1]
            y_scale = 1.0 + random.uniform(-0.1, 0.1)
            # x_scale, y_scale表示原图与结果图的倍数关系，如scale为2时缩小为原来的0.5
            matrix[0, 0] = x_scale
            matrix[1, 1] = y_scale
            if show_help_info:
                print("x_scale: ", x_scale, "  y_scale: ", y_scale)
        if do_rota:
            # 旋转（rotation）
            degree = np.random.randint(-15, 15)  # 旋转角度范围为[-15°,15°]
            radians = -np.pi * degree / 180.
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            matrix = np.dot(rotation, matrix)
            if show_help_info:
                print("degree: ", degree)

    elif dim == 3:
        if do_bspline:
            # B样条变换（B Spline）
            m=5
            spline_order = 3
            bspline = sitk.BSplineTransform(dim, spline_order)
            bspline.SetTransformDomainPhysicalDimensions(image.GetSize())
            mesh_size = [m, m, m]
            bspline.SetTransformDomainMeshSize(mesh_size)
            # Random displacement of the control points.
            # [13,18]为变形的强度，值越大变形越大
            originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * np.random.randint(13,18)
            bspline.SetParameters(originalControlPointDisplacements)
            image = resample(image, bspline)
        if do_flip:
            # 翻转（flip）
            if np.random.rand() > 0.5:
                image = image[::-1]  # 沿x轴反转
                if show_help_info:
                    print("x axis flip.")
            if np.random.rand() > 0.5:
                image = image[:, ::-1]  # 沿y轴反转
                if show_help_info:
                    print("y axis flip.")
            if np.random.rand() > 0.5:
                image = image[:, :, ::-1]  # 沿z轴反转
                if show_help_info:
                    print("z axis flip.")
        if do_trslt:
            # 平移（translation）
            x_trslt, y_trslt, z_trslt = np.random.randint(-20, 20, 3)  # 每个轴的平移范围在[-20,20]像素之间
            transform.SetTranslation((float(x_trslt), float(y_trslt), float(z_trslt)))
            if show_help_info:
                print("x_trslt: ", x_trslt, "  y_trslt: ", y_trslt, "  z_trslt: ", z_trslt)
        if do_scale:
            # 缩放（scale）
            x_scale = 1.0 + random.uniform(-0.1, 0.1)  # 缩放范围为原来的[0.9,1.1]
            y_scale = 1.0 + random.uniform(-0.1, 0.1)
            z_scale = 1.0 + random.uniform(-0.1, 0.1)
            # x_scale, y_scale, z_scale表示原图与结果图的倍数关系，如scale为2时缩小为原来的0.5
            matrix[0, 0] = x_scale
            matrix[1, 1] = y_scale
            matrix[2, 2] = z_scale
            if show_help_info:
                print("x_scale: ", x_scale, "  y_scale: ", y_scale, "  z_scale: ", z_scale)
        if do_rota:
            # 旋转（rotation）
            x_dgr, y_dgr, z_dgr = np.random.randint(-15, 15, 3)  # 旋转角度范围为[-15°,15°]
            x_rad = -np.pi * x_dgr / 180.
            y_rad = -np.pi * y_dgr / 180.
            z_rad = -np.pi * z_dgr / 180.
            rotation = np.array([[np.cos(y_rad) * np.cos(z_rad), np.cos(y_rad) * np.sin(z_rad), -np.sin(y_rad)],
                                 [-np.cos(x_rad) * np.sin(z_rad) + np.sin(x_rad) * np.sin(y_rad) * np.cos(z_rad),
                                  np.cos(x_rad) * np.cos(z_rad) + np.sin(x_rad) * np.sin(y_rad) * np.sin(z_rad),
                                  np.sin(x_rad) * np.cos(y_rad)],
                                 [np.sin(x_rad) * np.sin(z_rad) + np.cos(x_rad) * np.sin(y_rad) * np.cos(z_rad),
                                  -np.sin(x_rad) * np.cos(z_rad) + np.cos(x_rad) * np.sin(y_rad) * np.sin(z_rad),
                                  np.cos(x_rad) * np.cos(y_rad)]])
            matrix = np.dot(rotation, matrix)
            if show_help_info:
                print("x_dgr: ", x_dgr, "  y_dgr: ", y_dgr, "  z_dgr: ", z_dgr)

    transform.SetMatrix(matrix.ravel())
    # 以下两行是为了让图像的中心点为物体的中心点
    center = image.GetOrigin() + np.array(image.GetSize()) * np.array(image.GetSpacing()) * 0.5
    transform.SetCenter(center)
    out = resample(image, transform)

    # 保持和原图像相同的窗宽和窗位，同时做了归一化
    # 不加此步，则图像会在缩放和旋转的时候导致灰度值改变
    out_arr = sitk.GetArrayFromImage(out)
    min_window = float(window_center) - 0.5 * float(window_width)
    out_arr = (out_arr - min_window) / float(window_width)
    out_arr[out_arr < 0] = 0.
    out_arr[out_arr > 1] = 1.
    if dim == 2:
        return out_arr, dim
    elif dim == 3:
        return sitk.GetImageFromArray(out_arr), dim


if __name__ == "__main__":
    path_2D = r"C:\Users\zuzhiang\Desktop\1.jpg"
    path_3D = r"C:\Data\1.Work\02.Code\DDR\results\f_img.nii"
    out_path_3D = r"C:\Users\zuzhiang\Desktop\out.nii"
    # out, dim = data_augmentation(path_3D)
    out, dim = data_augmentation(image=sitk.ReadImage(path_3D))
    if dim == 2:
        plt.imshow(out)
        plt.title("result image")
        plt.show()
    elif dim == 3:
        sitk.WriteImage(out, out_path_3D)
    print("end")
