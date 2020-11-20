import os
import glob
import numpy as np
import SimpleITK as sitk


# 获取可以包裹mask的最小bounding box
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


# 根据bbox截取图片
def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    # slice是切片函数，参数为：起始值，终止值，[步长]
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


out_path = r"C:\Users\zuzhiang\Desktop"
files = glob.glob(os.path.join(r"C:\Data\1.Work\03.DataSet\LPBA40\train\256-256-256", "*.nii.gz"))
print("Number of images: ", len(files))

minx, miny, minz, maxx, maxy, maxz = 300, 300, 300, 0, 0, 0
for file in files:
    _, name = os.path.split(file)
    print("Name: ", name)
    old_img = sitk.ReadImage(file)
    img_arr = sitk.GetArrayFromImage(old_img)
    bbox = get_bbox_from_mask(img_arr)
    if bbox[0][0] < minx:
        minx = bbox[0][0]
    if bbox[0][1] > maxx:
        maxx = bbox[0][1]
    if bbox[1][0] < miny:
        miny = bbox[1][0]
    if bbox[1][1] > maxy:
        maxy = bbox[1][1]
    if bbox[2][0] < minz:
        minz = bbox[2][0]
    if bbox[2][1] > maxz:
        maxz = bbox[2][1]
    print("bbox: ", bbox)
print("\nminx: %d  maxx: %d  miny: %d  maxy: %d  minz: %d  maxz: %d" % (minx, maxx, miny, maxy, minz, maxz))

# # 根据以上6个值手工得到bbox
# bbox = [[0, 240], [21, 229], [40, 216]]
# print("bbox: ", bbox)
# for file in files:
#     _, name = os.path.split(file)
#     print("Name: ", name)
#     old_img = sitk.ReadImage(file)
#     img_arr = sitk.GetArrayFromImage(old_img)
#     img_arr = crop_to_bbox(img_arr, bbox)
#     print("new shape: ", img_arr.shape)
#     img = sitk.GetImageFromArray(img_arr)
#     img.SetOrigin(old_img.GetOrigin())
#     img.SetDirection(old_img.GetDirection())
#     img.SetSpacing(old_img.GetSpacing())
#     sitk.WriteImage(img, os.path.join(out_path, name))
