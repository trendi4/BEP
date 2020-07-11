### Requirements

* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

### How to use

The extracted patches will be located in the output_patches repository. The images the network will use to make the predictions are located in data/obj/ and are images of resolution 682x682. 

To run the extraction of the predictions use the following command:
```
python extract.py
```
In case you have TIF stacks, place them in tif_stacks repository and to extract the patches use:

```
python extract.py tif_to_jpg
```

In case you have 2048x682 images, place them in data/obj/ and use:

```
python extract.py crop
```

The default weights are those of Yolo-tiny-PRN. If you want to change it to YOLOv4-416 use:
```
python extract.py yolov4
```

The height of the prediction patches can be changed. By default it is set at 1.5 times the bouding box height. If you want to make it two times type:

```
python extract.py 2
```

Finally, these commands can all be combined. In case you want to extract predictions from .tif images, using Yolov4, with two times the bouding box height use:

```
python extract.py tif_to_jpg yolov4 2
```
