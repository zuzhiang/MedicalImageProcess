import os
import glob
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import models.data_augmentation as da

path = r"C:\Data\1.Work\03.DataSet\LPBA40"
out_path = r"C:\Data\1.Work\03.DataSet\LPBA40_DA"
#file_lst = glob.glob(os.path.join(path, '*/*bfc.nii'))
file_lst=[r"C:\Data\1.Work\02.Code\DDR\fixed_LPBA40.nii"]
num = len(file_lst)
# for file in file_lst:
#     img, dim = da.data_augmentation(file, do_bspline=False, do_trslt=True, do_scale=True, do_rota=True,
#                                     show_help_info=True)
#     _, name = os.path.split(file)
#     print("path: ", os.path.join(out_path, name))
#     sitk.WriteImage(img, os.path.join(out_path, name))

for file in file_lst:
    # img, dim = da.data_augmentation(file, do_bspline=False, do_trslt=False, do_scale=False, do_rota=False,
    # show_help_info=True)
    img = sitk.ReadImage(file)
    img_arr = sitk.GetArrayFromImage(img)
    new_shape = [256, 128, 256]
    new_img = resize(img_arr, new_shape, order=3, mode='constant', cval=0, preserve_range=bool)
    # order几次样条插值，cval外部补零，保留原数据（否则被默认标准化、归一化之类的乱七八糟）
    '''
    The order of interpolation. The order has to be in the range 0-5:
    0: Nearest-neighbor
    1: Bi-linear (default)
    2: Bi-quadratic
    3: Bi-cubic
    4: Bi-quartic
    5: Bi-quintic
    '''
    new_img = sitk.GetImageFromArray(new_img)
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    print("file: ", file)
    #temp_path = r"C:\Users\zuzhiang\Desktop\out.nii"
    sitk.WriteImage(new_img, file)
