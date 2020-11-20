import os
import glob

path = r"/home/syzhou/zuzhiang/Dataset/MGH10/Heads"
# 读取目录下的nii.gz文件
images = glob.glob(os.path.join(path,"*.img"))
# 下面为freesurfer的环境配置命令
a = "export FREESURFER_HOME=/home/syzhou/zuzhiang/freesurfer;"
b = "source $FREESURFER_HOME/SetUpFreeSurfer.sh;"
# 数据所在的目录
c = "export SUBJECTS_DIR="+path+";"

#images=['/home/syzhou/zuzhiang/Dataset/MGH10/Heads/1127.img']
for image in images:
    # 将文件路径和文件名分离
    filename = os.path.split(image)[1] # 将路径名和文件名分开
    filename = os.path.splitext(filename)[0] #将文件名和扩展名分开，如果为.nii.gz，则认为扩展名是.gz
    # freesurfer环境配置、颅骨去除、未仿射对齐mpz转nii、仿射对齐、仿射对齐mpz转nii.gz格式
    #recon-all是颅骨去除的命令
    # mri_convert是进行格式转换，从mgz转到nii.gz，只是为了方便查看
    # --apply_transform：仿射对齐操作
    # 转格式
    filename=filename[:] #根据扩展名的不同，这里需要做更改，只保留文件名即可
    cur_path=os.path.join(path,filename) 
    print("file name: ",cur_path)
    cmd = a + b + c \
          + "recon-all -parallel -i " + image + " -autorecon1 -subjid " + cur_path + "&&" \
          + "mri_convert " +  cur_path + "/mri/brainmask.mgz " +cur_path + "/mri/"+filename+".nii.gz;"\
          + "mri_convert " + cur_path + "/mri/brainmask.mgz --apply_transform " + cur_path + "/mri/transforms/talairach.xfm -o " + cur_path + "/mri/brainmask_affine.mgz&&" \
          + "mri_convert " + cur_path + "/mri/brainmask_affine.mgz " + cur_path + "/mri/"+filename+"_affine.nii.gz;"
    #print("cmd:\n",cmd)
    os.system(cmd)