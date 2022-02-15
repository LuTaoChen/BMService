# 一些模型转换

## VGG16和ResNet50 （tensorflow slim版本）

### 来源
```shell
git clone https://github.com/tensorflow/models.git
cd models/research/slim
```
### 数据集

ImageNet2012验证集，到ImageNet官网，注册帐号，下载2012年挑战赛版本

### 生成vgg16.pb和resnet50.pb
根据里面的export_inference_graph.py生成vgg_16.pb、resnet_v1_50.pb，例如
```shell
python export.py --model_name vgg_16 --check_point vgg_16.ckpt --labels_offset 1 --output_names "vgg_16/fc8/squeezed"
python export.py --model_name resnet_v1_50 --check_point resnet_v1_50.ckpt --labels_offset 1 --output_names "resnet_v1_50/predictions/Reshape_1"
```
### bmodel转换

准备图片目录dataset, 转换为image_lmdb, vgg16和resnet50都可以使用
```shell
pushd dataset
ls *.JPEG >imagelist.txt
pop
convert_imageset dataset/ dataset/imagelist.txt image_lmdb
```
-----------------------------------------------------------------------------------
#### vgg16:
生成fp32 bmodel

```shell
python3 -m bmnett --model vgg_16.pb --input_names "input" --output_names "vgg_16/fc8/squeezed" --shapes "[[1,224,224,3]]" --target BM1684 --v 4 --enable_profile 1 --opt 2 2>&1 --outdir vgg16_fp32_bmodel | tee vgg16.log
```

生成fp32 umodel

```shell
export UMODEL_DIR=vgg16_umodel
python3 -m bmnett --mode GenUmodel --model vgg_16.pb --input_names "input" --output_names "vgg_16/fc8/squeezed" --shapes "[[1,224,224,3]]" --target BM1684 --v 4 --enable_profile 1 --opt 2 2>&1 --outdir $UMODEL_DIR | tee vgg16_umodel.log
```

量化

```shell
python3 -m ufw.tools.to_umodel -u $UMODEL_DIR/bm_network_bmnett.fp32umodel -p VGG -D /workspace/model/image_lmdb
calibration_use_pb quantize -model $UMODEL_DIR/bm_network_bmnett_test_fp32.prototxt -weights $UMODEL_DIR/bm_network_bmnett.fp32umodel -iterations=12 --debug_log_level=2 -bitwidth=TO_INT8
```

生成fix8b bmodel

```shell
bmnetu --weight $UMODEL_DIR/bm_network_bmnett.int8umodel --model $UMODEL_DIR/bm_network_bmnett_deploy_int8_unique_top.prototxt --v 4 --outdir vgg16_fix8b_bmodel --max_n 4 2>&1 | tee vgg16_fix8b.log
```
--------------------------------------------------------------------------------------

#### resnet50

生成fp32 bmodel

```shell
python3 -m bmnett --model resnet_v1_50.pb --input_names "input" --output_names "resnet_v1_50/predictions/Reshape_1" --shapes "[[1,224,224,3]]" --target BM1684 --v 4 --enable_profile 1 --opt 2 2>&1 --outdir resnet50_fp32_bmodel | tee resnet50_fp32.log
```

生成fp32 umodel

```shell
export UMODEL_DIR=resnet50_umodel
python3 -m bmnett --mode GenUmodel --model resnet_v1_50.pb --input_names "input" --output_names "resnet_v1_50/predictions/Reshape_1" --shapes "[[1,224,224,3]]" --target BM1684 --v 4 --enable_profile 1 --opt 2 2>&1 --outdir $UMODEL_DIR | tee resnet50_umodel.log
```

量化

```shell
python3 -m ufw.tools.to_umodel -u $UMODEL_DIR/bm_network_bmnett.fp32umodel -p VGG -D /workspace/model/image_lmdb
calibration_use_pb quantize -model $UMODEL_DIR/bm_network_bmnett_test_fp32.prototxt -weights $UMODEL_DIR/bm_network_bmnett.fp32umodel -iterations=12 --debug_log_level=2 -bitwidth=TO_INT8
```

生成fix8b bmodel

```shell
bmnetu --weight $UMODEL_DIR/bm_network_bmnett.int8umodel --model $UMODEL_DIR/bm_network_bmnett_deploy_int8_unique_top.prototxt --v 4 --outdir resnet50_fix8b_bmodel --max_n 4 2>&1 | tee resnet50_fix8b.log
```

## bert模型编译及运行（tensorflow1.x）

### bert来源
https://github.com/google-research/bert/
下载 https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
### 数据集
squad

复制数据集 squad到bert目录下
将uncased_L-12_H-768_A-12.zip复制到bert目录下并解压

### finetune

```shell
export BERT_BASE_DIR=uncased_L-12_H-768_A-12
export SQUAD_DIR=squad
python3 run_squad.py \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--do_train=True \
--train_file=$SQUAD_DIR/train-v1.1.json \
--do_predict=True \
--predict_file=$SQUAD_DIR/dev-v1.1.json \
--train_batch_size=8 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir=squad_out
```

完成后会在squad_out生成model.ckpt-21899*系列文件

### 生成离线模型

```shell
export MAX_BATCH=1
```
将freeze_squad.py文件放到bert目录中

```shell
python freeze_squad.py --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint squad_out/model.ckpt-21899 --batch_size $MAX_BATCH --output_name squad_graph.pb
```

这里的model.ckpt-21899要根据finetune结果选择，会在当前目录下生成squad_graph.pb


### 转成 fp32 bmodel

```shell
python3 -m bmnett --model squad_graph.pb --target BM1684 --shapes="[$MAX_BATCH,384], [$MAX_BATCH,384], [$MAX_BATCH,384]" --input_names="input_ids, input_mask, segment_ids" --descs="[0,int32,0,256],[1, int32, 0, 2], [2, int32, 0, 2]" --outdir squad_bmodel --v=4 --cmp=True
```

## DeepLabv3 Tensorflow model zoo 版本

### 来源
https://github.com/tensorflow/models/blob/f670e89c47573d1f0465bd1a20b4c36dae064be1/research/deeplab/g3doc/model_zoo.md
下载里面的pb模型，例如
```shell
wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz
```
### 数据集
VOC2012数据集，其中数据集目录结构为
```shell
VOC2012
|-- Annotations
|-- ImageSets
|   |-- Action
|   |-- Layout
|   |-- Main
|   `-- Segmentation
|-- JPEGImage
|-- SegmentationClass
|-- SegmentationObject

```
把分割任务用的验证集ImageSets/Segmentation/val.txt里面的图片提取出来放到一个文件夹内，参考tool/extract_voc.py，修改脚本内的路径参数

得到的文件夹用来验证

### fp32模型编译
```shell
python3 -m bmnett --model "frozen_inference_graph.pb" --input_names "ImageTensor" --shapes "[1,513,513,3]" --desc="[0,uint8,0,256]" --target BM1684 --opt 1 --cmp False --outdir deeplabv3_tf_fp32 --v 4
```

 ------------------------------------------------------------------------------------------------
## After finish the bmodels' compilation put then into the structure like

```shell
|-- bert_squad
|   `-- fp32.bmodel
|-- inception
|   `-- fix8b.bmodel
|-- resnet101
|   `-- fix8b_4n.bmodel
|-- resnet50_v1
|   |-- fix8b.bmodel
|   `-- fix8b_4n.bmodel
|-- vgg16
|   `-- fix8b.bmodel
|-- yolov3
|   `-- fix8b_b4.bmodel
|-- yolov5s
|  `-- yolov5s_1b_fp32.bmodel
`-- yolov5x
    `-- yolov5x_1b_int8.bmodel
```
