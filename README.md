## Image reprojection

In [Monodepth2](https://arxiv.org/abs/1806.01260), the reprojection loss was introduced. However, it was applied only for **Perspective cameras**. 

#### Reprojection loss:
```math
L_{reproj}=\frac{1}{l}\sum_{i=1}^{l}\|I_t-I_{t'} \left< proj\left<(D_t,RT_{t\rightarrow t'},K) \right> \right)\|_1,
```
where $I_t$ and $I_{t'}$ are the target and the source views, $D_t$ is the depth map, $RT$ is the transformation between $t$ and $t'$ moments, K is the intrinsics, and $\left<\cdot\right>$ is the sampling operator.

In this repo, the extension for **Fisheye cameras** is introduced. $\left<\cdot\right>$ methods for fisheyes are located in  _reprojection.fisheye_reproj_. Examples of reprojection for perspective and fisheye cameras based on the [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes_aligned/) are presented in _example.ipynb_.