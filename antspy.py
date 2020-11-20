import os
import glob
import ants
import numpy as np
import SimpleITK as sitk

# ants图片的读取
f_img = ants.image_read("./data/f_img.nii.gz")
m_img = ants.image_read("./data/m_img.nii.gz")
f_label = ants.image_read("./data/f_label.nii.gz")
m_label = ants.image_read("./data/m_label.nii.gz")

'''
ants.registration()函数的返回值是一个字典：
    warpedmovout: 配准到fixed图像后的moving图像 
    warpedfixout: 配准到moving图像后的fixed图像 
    fwdtransforms: 从moving到fixed的形变场 
    invtransforms: 从fixed到moving的形变场

type_of_transform参数的取值可以为：
    Rigid：刚体
    Affine：仿射配准，即刚体+缩放
    ElasticSyN：仿射配准+可变形配准，以MI为优化准则，以elastic为正则项
    SyN：仿射配准+可变形配准，以MI为优化准则
    SyNCC：仿射配准+可变形配准，以CC为优化准则
'''
# 图像配准
mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')
# 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                   interpolator="linear")
# 对moving图像对应的label图进行配准
warped_label = ants.apply_transforms(fixed=f_img, moving=m_label, transformlist=mytx['fwdtransforms'],
                                     interpolator="linear")
# 将配准后图像的direction/origin/spacing和原图保持一致
warped_img.set_direction(f_img.direction)
warped_img.set_origin(f_img.origin)
warped_img.set_spacing(f_img.spacing)
warped_label.set_direction(f_img.direction)
warped_label.set_origin(f_img.origin)
warped_label.set_spacing(f_img.spacing)
img_name = "./result/warped_img.nii.gz"
label_name = "./result/warped_label.nii.gz"
# 图像的保存
ants.image_write(warped_img, img_name)
ants.image_write(warped_label, label_name)

# 将antsimage转化为numpy数组
warped_img_arr = warped_img.numpy(single_components=False)
# 从numpy数组得到antsimage
img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False)
# 生成图像的雅克比行列式
jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False, geom=False)
ants.image_write(jac, "./result/jac.nii.gz")
# 生成带网格的moving图像，实测效果不好
m_grid = ants.create_warped_grid(m_img)
m_grid = ants.create_warped_grid(m_grid, grid_directions=(False, False), transform=mytx['fwdtransforms'],
                                 fixed_reference_image=f_img)
ants.image_write(m_grid, "./result/m_grid.nii.gz")

'''
以下为其他不常用的函数：

ANTsTransform.apply_to_image(image, reference=None, interpolation='linear')
ants.read_transform(filename, dimension=2, precision='float')
# transform的格式是".mat"
ants.write_transform(transform, filename)
# field是ANTsImage类型
ants.transform_from_displacement_field(field)
'''

print("End")
