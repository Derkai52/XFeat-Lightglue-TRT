# XFeat + LightGlue TensorRT 部署

## 环境
- Nvidia JetPack 5.1.3
- CUDA 11.4
- TensorRT 8.5.2
- OpenCV 4.X
- PyTorch 2.1.0

## 下载并编译
``` bash
git clone https://github.com/Derkai52/XFeat-Lightglue-TRT.git

cd XFeat-Lightglue-TRT

mkdir build
cd build
cmake ..
make
```

## 导出ONNXmox
```bash
cd scripts
# 默认导出是 输入 800x800x1 灰度图像，最大512个匹配对，这里可以按照需要进去修改
python3 export.py
# 会导出一个 Xfeat的引擎和LIghtglue的引擎
```


## ONNX 转为 Engine
```bash
/usr/local/TensorRT-8.6.1.6/bin/trtexec --onnx=/home/xxx/XFeat-Lightglue-TRT/weights/lighterglue_L3.onnx --saveEngine=/home/xxx/XFeat-Lightglue-TRT/weights/lighterglue_L6.engine
```

## Demo 演示
```bash
cd build
./match_test 
```