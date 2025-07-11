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

## 导出 ONNX 模型
```bash
# 以下是示例 输入 800x800x1 灰度图像，最大512个匹配对，这里可以按照需要进去修改

# 导出 Xfeat.onnx 文件
python3 scripts/export.py --xfeat_only_model --height 800 --width 800 --top_k 512 --split_instance_norm --export_path ./weights/xfeat_1_800_800.onnx
# 导出 Lightglue.onnx 文件
python3 scripts/export.py --xfeat_only_lighterglue --height 800 --height 800 --top_k 512 --export_path ./weights/lightglue_L6_1_800_800.onnx
```


## ONNX 转为 Engine
```bash
# 如果需要INT8 推理的话，可以在后面添加 --int8 参数，默认是FP32推理
# 转为 Xfeat.engine 文件
/usr/src/tensorrt/bin/trtexec --onnx=/home/emnavi/GNSS-Denial-UAV-Location/src/match_location/weights/xfeat_1_800_800.onnx --saveEngine=/home/tk/GNSS-Denial-UAV-Location/src/match_location/weights/xfeat_1_800_800.engine

# 转为 Lightglue.engine 文件
/usr/src/tensorrt/bin/trtexec --onnx=/home/emnavi/GNSS-Denial-UAV-Location/src/match_location/weights/lightglue_L6_1_800_800.onnx --saveEngine=/home/emnavi/GNSS-Denial-UAV-Location/src/match_location/weights/lightglue_L6_1_800_800.engine
```

## Demo 演示
```bash
cd build
./match_test 
```