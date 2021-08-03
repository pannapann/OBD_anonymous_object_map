# Anonymous Object Detection and Mapping (AODM)

Anonymous object detection with depth detection with pseudolidar pointcloud from a single image

## Installation
First create conda environment from requirements.txt (Python 3.6.6 is our test env)

```bash
conda create --name <envname> --file requirements.txt
```

Then install pypi dependent library from pipreqs.txt after activating the created environment

```bash
pip install -r pipreqs.txt
```

After that install pytorch with cuda from https://pytorch.org/

Then replace the pyntcloud library file, pyvista_backend.py in the downloaded library to the file given in this repo (/asset) with the same name.

Or use the environment.yml provided 

## Usage

### 1. Anonymous object detection

```bash
python predict_bounded.py --video <video_path> --monodepth2_model_name HR_Depth_K_M_1280x384 --pred_metric_depth
```

<p align="center">
  <img src="asset/predict_bounded_result.gif" alt="Example prediction output" width="1280" />
</p>

[Full Anonymous object detection video link](https://www.youtube.com/watch?v=zVKHdSpBkBE)
### 2. Pseudo Lidar

```bash
python predict_pointcloud.py --video <video_path> --monodepth2_model_name HR_Depth_K_M_1280x384 --pred_metric_depth
```

<p align="center">
  <img src="asset/pointcloud_demo.gif" alt="Example pointcloud output" width="1280" />
</p>

[Full Pseudo Lidar video link](https://www.youtube.com/watch?v=-W8eXJR-gM4)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Github used during development
[Monodepth2](https://github.com/nianticlabs/monodepth2)

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

[HR-Depth](https://github.com/shawLyu/HR-Depth)


> **HR-Depth: High Resolution Self-Supervised Monocular Depth Estimation**
>
> Xiaoyang Lyu, Liang Liu, Mengmeng Wang, Xin Kong, Lina Liu, Yong Liu*, Xinxin Chen and Yi Yuan.

[Pseudo-Lidar](https://github.com/mileyan/pseudo_lidar)

> **Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving**
>
> Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell and Kilian Q. Weinberger


## License
[MIT](https://choosealicense.com/licenses/mit/)
