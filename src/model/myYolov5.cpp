/**
 * @file myYolov5.cpp
 * @author your name (you@domain.com)
 * @brief as long as out put shape = [b, c * (cls + 5), h, w]
 * @version 0.1
 * @date 2022-01-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <dirent.h>
#include<vector>
#include<map>
#include<thread>
#include<sys/stat.h>
#include <algorithm>
#include <fstream>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "BMDetectUtils.h"
#include "bmcv_api.h"
#include <type_traits>

using namespace bm;
#define OUTPUT_RESULT_FILE  "yolov5_result.json"
std::map<size_t, std::string> globalLabelMap;
std::map<std::string, size_t> globalImageIdMap;
std::map<size_t, size_t> categoryInCoco;
std::map<std::string, std::vector<DetectBox>> globalGroundTruth;

struct YOLOv5Config {
    bool initialized = false;
    bool isNCHW;
    int anchors = 3;
    int detHeads = 3;
    int classNum = 80;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;

    // use static to cache resizedImage, to avoid allocating memory everytime
    std::vector<bm_image> resizedImages;
    std::vector<bm_image> tempImg;
    // bmcv_image_convert_to do not support RGB_PACKED format directly
    // use grayImages as a RGB_PACKED wrapper
    std::vector<bm_image> grayImages;
    // used as a wrapper of input tensor
    std::vector<bm_image> preOutImages;
    std::string savePath = "yolov5_out";
    std::vector<std::vector<std::vector<int>>> m_anchors{{{10, 13}, {16, 30}, {33, 23}},
                                                        {{30, 61}, {62, 45}, {59, 119}},
                                                        {{116, 90}, {156, 198}, {373, 326}}};

    float probThreshold;
    float iouThreshold;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        std::string command;
        command = "mkdir -p " + savePath;  
        system(command.c_str());
        ctx->setConfigData(this);
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
        netHeight = inTensor->shape(2);
        netWidth = inTensor->shape(3);
        netFormat = FORMAT_RGB_PLANAR; // for NHWC input
        float input_scale = 1.0;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32; 
            probThreshold = 0.5;
            iouThreshold = 0.5;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            probThreshold = 0.001;
            iouThreshold = 0.5;
            input_scale = inTensor->get_scale();
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }
        float scale = 1.0/255;
        float bias = 0;
        float real_scale = scale * input_scale;
        float real_bias = bias * input_scale;
        ConvertAttr.alpha_0 = real_scale;
        ConvertAttr.beta_0 = real_bias;
        ConvertAttr.alpha_1 = real_scale;
        ConvertAttr.beta_1 = real_bias;
        ConvertAttr.alpha_2 = real_scale;
        ConvertAttr.beta_2 = real_bias;

        resizedImages = ctx->allocAlignedImages(netBatch, netHeight, netWidth, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
        if(!isNCHW){
            grayImages = ctx->allocImagesWithoutMem(
                        netBatch, netHeight, netWidth*3, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, 64);
            bm_device_mem_t resizedMem;
            bm_image_get_contiguous_device_mem(resizedImages.size(), resizedImages.data(), &resizedMem);
            bm_image_attach_contiguous_mem(grayImages.size(), grayImages.data(), resizedMem);
            preOutImages = ctx->allocImagesWithoutMem(
                        netBatch, netHeight, netWidth*3, FORMAT_GRAY, netDtype);
        } else {
            preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
        }
    }
};


using InType = std::vector<std::string>;

struct PostOutType {
    std::vector<std::string> rawIns;
    std::vector<std::vector<DetectBox>> results;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    thread_local static YOLOv5Config cfg;
    cfg.initialize(inTensor, ctx);

    std::vector<bm_image> * alignedInputs = new std::vector<bm_image>;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs->push_back(image);
    }        
    bmcv_color_t color = {114, 114, 114};

    aspectScaleAndPad(ctx->handle, *alignedInputs, cfg.resizedImages, color);
    saveImage(cfg.resizedImages[0], "resize.jpg");

    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

    if(cfg.isNCHW){
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.resizedImages.data(), cfg.preOutImages.data());
    } else {
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.grayImages.data(), cfg.preOutImages.data());
    }
    // pass input info to post process
    ctx->setPreExtra(alignedInputs);
    return true;
}

struct CoordConvertInfo {
    size_t inputWidth;
    size_t inputHeight;
    float ioRatio;
    float oiRatio;
    float hOffset;
    float wOffset;
};

template<typename T1, typename T2, typename T3>
bool yoloV5BoxParse(DetectBox& box, 
                    T1 *data_xywh,
                    T2 *data_obj,
                    T3 *data_cls,
                    float factor_xywh,
                    float factor_obj,
                    float factor_cls,
                    int c_stride,
                    size_t tensor_idx,
                    size_t anchor_idx,
                    size_t grid_idx,
                    size_t grid_w,
                    size_t grid_h,
                    YOLOv5Config cfg){

    float objectness = *((T2 *)data_obj) * factor_obj;
    if (objectness < -log(1/(cfg.probThreshold) - 1)){ // objectness < 0 === sigmoid(obj) < 0.5
        return false;
    }
    int category = argmax(data_cls, cfg.classNum, c_stride);
    float cls_prob = (data_cls[category * c_stride]) * factor_cls;    
    // float confidence = sigmoid(objectness) * sigmoid(cls_prob);
    float confidence = sigmoid(objectness);
    if (objectness < cfg.probThreshold){ 
        return false;
    }
    float x = *(data_xywh) * factor_xywh;
    float y = *(data_xywh + c_stride) * factor_xywh;
    float w = *(data_xywh + 2 * c_stride) * factor_xywh;
    float h = *(data_xywh + 3 * c_stride) * factor_xywh;
    box.confidence = confidence;
    box.category = categoryInCoco[category];;
    box.categoryName = globalLabelMap[category];

    x = (sigmoid(x) * 2 + grid_idx % grid_w - 0.5) * cfg.netWidth / grid_w;
    y = (sigmoid(y) * 2 + grid_idx / grid_w - 0.5) * cfg.netWidth / grid_h;
    w = pow(sigmoid(w) * 2, 2) * cfg.m_anchors[tensor_idx][anchor_idx][0];
    h = pow(sigmoid(h) * 2, 2) * cfg.m_anchors[tensor_idx][anchor_idx][1];
    box.xmin  = x - w / 2;
    box.ymin  = y - h / 2;
    box.xmax  = x + w / 2;
    box.ymax  = y + h / 2;

    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.rawIns = rawIn;
    if(rawIn.empty()) return false;
    auto pCfg = (YOLOv5Config*)ctx->getConfigData();
    auto& cfg = *pCfg;

    auto pInputImages = reinterpret_cast<std::vector<bm_image>*>(ctx->getPostExtra());
    auto realBatch = rawIn.size();
    auto outTensor = outTensors[0];
    size_t batch = outTensor->shape(0);
    std::vector<size_t> boxNums(outTensors.size());
    size_t totalBoxNum=0;
    auto singleDataSize = outTensor->shape(4);
    BM_ASSERT_EQ(cfg.classNum, cfg.classNum);
    BM_ASSERT_EQ(batch, pInputImages->size());

    std::vector<CoordConvertInfo> coordInfos(batch);
    for(size_t b=0; b<realBatch; b++){
        auto& image = (*pInputImages)[b];
        auto& ci = coordInfos[b];

        ci.inputWidth = image.width;
        ci.inputHeight = image.height;
        ci.ioRatio = std::max((float)ci.inputWidth/cfg.netWidth,
                               (float)ci.inputHeight/cfg.netHeight);
        ci.oiRatio = 1/ci.ioRatio;
        ci.hOffset = (cfg.netHeight - ci.oiRatio * ci.inputHeight)/2;
        ci.wOffset = (cfg.netWidth - ci.oiRatio * ci.inputWidth)/2;
    }
    for(size_t i = 0; i < cfg.detHeads; i++){
        auto tensor = outTensors[i * 3 + 1]; // cls
        BM_ASSERT_EQ(batch, tensor->shape(0));
        boxNums[i] = tensor->partial_shape_count(1,3);
        totalBoxNum += boxNums[i];
    }
    std::vector<std::vector<DetectBox>> batchBoxInfos(batch, std::vector<DetectBox>(totalBoxNum));

    // fill batchBoxInfo
    std::vector<int> batchIndice(batch, 0);
    for(size_t head_id = 0; head_id < cfg.detHeads; head_id++){
        uchar *xywh = outTensors[head_id]->get_raw_data();
        uchar *obj = outTensors[head_id + 3]->get_raw_data();
        uchar *cls = outTensors[head_id + 6]->get_raw_data();
        float factor_xywh = outTensors[head_id]->get_scale();
        float factor_obj = outTensors[head_id + 3]->get_scale();
        float factor_cls = outTensors[head_id + 6]->get_scale();
        auto xywh_t = outTensors[head_id * 3]->get_shape();
        int obj_t = outTensors[head_id * 3 + 1]->get_dtype();
        int cls_t = outTensors[head_id * 3]->get_dtype();
        size_t grid_w = outTensors[head_id]->shape(3);
        size_t grid_h = outTensors[head_id]->shape(2);
        auto boxNum = boxNums[head_id]; // 3 * 80 * 80, 3 * 40 * 40, 3 * 20 * 20
        
        int c_stride = grid_h * grid_w;
        int a_stride_xywh = 4 * c_stride;
        int a_stride_obj = 1 * c_stride;
        int a_stride_cls = cfg.classNum * c_stride;
        int b_stride_xywh = cfg.anchors * a_stride_xywh;
        int b_stride_obj = cfg.anchors * a_stride_obj;
        int b_stride_cls = cfg.anchors * a_stride_cls;
        
        for(size_t b=0; b<batch; b++){
            auto& ci = coordInfos[b];   
            for(int anchor = 0; anchor < cfg.anchors; anchor++){         
                for(int i = 0; i < c_stride; i++){
                    auto *box_x = (int8_t *)xywh + b * b_stride_xywh + anchor * a_stride_xywh + i;
                    auto *box_obj = (int8_t *)obj + b * b_stride_obj + anchor * a_stride_obj + i;
                    auto *box_cls = (int8_t *)cls + b * b_stride_cls + anchor * a_stride_cls + i;
                    
                    auto& boxInfo = batchBoxInfos[b][batchIndice[b]];
                    if(yoloV5BoxParse(boxInfo,
                                        box_x,
                                        box_obj,
                                        box_cls,
                                        factor_xywh,
                                        factor_obj,
                                        factor_cls,
                                        c_stride,
                                        head_id,
                                        anchor, 
                                        i,
                                        grid_w, 
                                        grid_h,
                                        cfg)){
                        batchIndice[b]++;
                    }
                }
            }
        }
    }
    for(size_t b=0; b<batch; b++){
        batchBoxInfos[b].resize(batchIndice[b]);
    }

    // final results
    postOut.results = batchNMS(batchBoxInfos, cfg.iouThreshold);
    // postOut.results = batchBoxInfos;

    for(size_t b=0; b<batch; b++){
        auto name = baseName(rawIn[b]);
        auto imageId = globalImageIdMap[name];
        auto& ci = coordInfos[b];
        for(auto& r: postOut.results[b]) {
            r.imageId = imageId;
            r.xmin = (r.xmin - ci.wOffset) * ci.ioRatio;
            r.xmax = (r.xmax - ci.wOffset) * ci.ioRatio;
            r.ymin = (r.ymin - ci.hOffset) * ci.ioRatio;
            r.ymax = (r.ymax - ci.hOffset) * ci.ioRatio;
        }
    }

    for(size_t b=0; b<batch; b++){
        auto name = baseName(rawIn[b]);
        drawDetectBoxEx(pInputImages->at(b), postOut.results[b], globalGroundTruth[name], cfg.savePath+"/"+name);
    }

    // clear extra data
    for(size_t i=0; i<pInputImages->size(); i++) {
        bm_image_destroy(pInputImages->at(i));
    }
    delete pInputImages;
    return true;
}

bool resultProcess(const PostOutType& out, std::vector<DetectBox>& allPredictions){
    if(out.rawIns.empty()) return false;
    auto batch = out.rawIns.size();
    for(auto b=0; b<batch; b++){
        auto name = baseName(out.rawIns[b]);
        BMLOG(INFO, "'%s' result", name.c_str());
        for(auto& box: out.results[b]){
            auto label = std::to_string(box.category);
            if(box.categoryName != ""){
                label += "-" + box.categoryName;
            }
            label += ":" + std::to_string(box.confidence);
            BMLOG(INFO, "  box [%d, %d, %d, %d], %s",
                  (size_t)box.xmin, (size_t)box.ymin, (size_t)box.xmax, (size_t)box.ymax, label.c_str());
        }
        allPredictions.insert(allPredictions.end(), out.results[b].begin(), out.results[b].end());
    }
    return true;
}

int main(int argc, char* argv[]){
    set_env_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/coco/images";
    // std::string dataPath = "/workspace/my/dataset/coco/images/oneImg";
    std::string bmodel = topDir + "models/yolov5s/myYolov5s_int8_bs4.bmodel";
    std::string refFile = topDir+ "data/coco/instances_val2017.json";
    std::string labelFile = topDir + "data/coco/coco_val2017.names";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];

    std::vector<DetectBox> allPredictions;
    globalGroundTruth =  readCocoDatasetBBox(refFile);

    globalLabelMap = loadLabels(labelFile);
    std::map<std::string, size_t> categoryToId;
    readCocoDatasetInfo(refFile, globalImageIdMap, categoryToId);
    for(auto& idLabel: globalLabelMap){
        categoryInCoco[idLabel.first] = categoryToId[idLabel.second];
        BMLOG(INFO, "%d->%d: %s", idLabel.first, categoryToId[idLabel.second], idLabel.second.c_str());
    }
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info(bmodel);
    info.start();
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const std::vector<std::string> names){
            return runner.push(names);
        });
        while(!runner.allStopped()){
            if(runner.canPush()) {
                runner.push({});
            } else {
                std::this_thread::yield();
            }
        }
    });
    std::thread resultThread([&runner, &info, &allPredictions](){
        PostOutType out;
        std::shared_ptr<ProcessStatus> status;
        bool stopped = false;
        while(true){
            while(!runner.pop(out, status)) {
                if(runner.allStopped()) {
                    stopped = true;
                    break;
                }
                std::this_thread::yield();
            }
            if(stopped) break;
            info.update(status, out.rawIns.size());
            if(!resultProcess(out, allPredictions)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    saveCocoResults(allPredictions, OUTPUT_RESULT_FILE);
    return 0;
}

