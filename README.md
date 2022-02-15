# BMService

BMService is a framework to pipeline the whole process include pre-process, forward, post-forward running on multiple BM168x chips


## 编译可执行程序

Clone this repository

``` shell
cd BMService
git submodule update --init --recursive
```

After initializing your bmnnsdk environment, run the following commands to build:

``` shell
mkdir build && cd build
cmake ..
make -j
```

BMSerice-xxx will be generated for running

## 运行
cd BMService/build

./BMService-resnet [PATH TO IMAGES] [PATH_TO_BMODEL]  [PATH_TO_VALTXT] [PATH_TO_LABEL]

./BMService-yolov5 [PATH TO IMAGES] [PATH_TO_BMODEL] [PATH_TO_LABEL] [PATH_TO_COCONAMES]

./BMService-deeplabv3_tf [PATH TO IMAGES] [PATH_TO_BMODEL] [PATH_TO_LABEL]

## 模型编译
 [COMPILATION.md](./COMPILATION.md)

## <div align="center">其它</div>

## 测量COCO数据集的mAP

python3 ../tool/calc_map.py --anno ../data/coco/instances_val2017.json --log $(RESULTJSON) --image-dir $(IMAGEPATH)


## BMService的python环境准备和bert运行

下载最新的BMService代码

```shell
cd BMService
mkdir build && cd build && cmake ../ && make -j && cd ..

mkdir -p python/bmservice/lib
cp build/libbmservice.so python/bmservice/lib
export PYTHONPATH=`pwd`/python
```

将BMservice/examples/bert/bmservice_squad.py复制到finetune的bert目录下, 修改里面的脚本参数：bmodel_path和input_file
在bert目录下

```shell
python3 bmservice_squad.py
```

bert计算精度

```shell
python3 squad/evaluate-v1.1.py --dataset_file squad/dev-v1.1.json --prediction_file squad_eval_out/prediction.json
```
