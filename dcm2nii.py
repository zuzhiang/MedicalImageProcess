import os
import SimpleITK as sitk


def dcm2nii(dcm, nii):
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm, series_id[0])
    print(len(series_file_names))
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    print("type: ",type(image3d))
    sitk.WriteImage(image3d, nii)


if __name__=="__main__":
    '''
    dcm对应的文件夹下有很多子文件夹，每个子文件夹是一套dicom图像，将所有的dicom
    图像转换为.nii格式的图像
    '''
    dcm=input("dcm_dir:\n")
    nii = input("nii_dir:\n")
    for root,dirs,files in os.walk(dcm):
        print(dirs)
        break
    for dir in dirs:
        dcm_dir=dcm+"\\"+dir
        if not os.listdir(dcm_dir): #若文件夹为空则不处理
            continue
        nii_file=nii+"\\"+dir+".nii" 
        dcm2nii(dcm_dir,nii_file) #将dicom转为nii