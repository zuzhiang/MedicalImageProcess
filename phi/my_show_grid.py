import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# 空间转换网络
class SpatialTransformer(nn.Module):
    # 1.生成网格grid；2.new_grid=grid+flow，即旧网格加上一个位移；3.将网格规范化到[-1,1]；4.根据新网格对原图进行采样
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)  # 维度置换，变为0,2,3,1
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


# 生成网格图片
def create_grid(size, path):
    num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path)  # 保存图像
    # plt.show()


if __name__ == "__main__":
    out_path = r"C:\Users\zuzhiang\Desktop\new_img.nii"  # 图片保存路径
    # 读入形变场
    phi = sitk.ReadImage("./2D1.nii")  # [324,303,2]
    phi_arr = torch.from_numpy(sitk.GetArrayFromImage(phi)).float()
    phi_shape = phi_arr.shape
    # 产生网格图片
    create_grid(phi_shape, out_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))[..., 0]
    img = np.squeeze(img)[np.newaxis, np.newaxis, :phi_shape[0], :phi_shape[1]]
    # 用STN根据形变场对网格图片进行变形
    STN = SpatialTransformer(phi_shape[:2])
    phi_arr = phi_arr.permute(2, 0, 1)[np.newaxis, ...]
    warp = STN(torch.from_numpy(img).float(), phi_arr)
    # 保存图片
    warp_img = sitk.GetImageFromArray(warp[0, 0, ...].numpy().astype(np.uint8))
    sitk.WriteImage(warp_img, out_path)
    print("end")
