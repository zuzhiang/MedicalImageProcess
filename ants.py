import os
import glob

path="/home/syzhou/zuzhiang/Dataset/LPBA40"
out_path="/home/syzhou/zuzhiang/MIP/ANTs"
f_name="/home/syzhou/zuzhiang/Dataset/LPBA40/1.nii.gz"
for i in range(2,41):
    m_name=os.path.join(path,str(i)+".nii.gz")
    out_name=str(i)+"m"
    cmd= "antsRegistrationSyN.sh -d 3 -f " + f_name + " -m " + m_name + " -o " + out_name
    print("cmd: ",cmd)
    os.system(cmd)
print("End")
