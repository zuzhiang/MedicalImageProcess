import numpy as np
import SimpleITK as sitk
import nibabel as nib
import cv2

np.set_printoptions(threshold=np.inf)
input_filename = "./test.img"
img=sitk.ReadImage(input_filename)
#将图片转化为数组
img_arr=sitk.GetArrayFromImage(img)
out = sitk.GetImageFromArray(img_arr)
out.SetDirection(img.GetDirection())
sitk.WriteImage(out, "./out.nii")
print("end")
