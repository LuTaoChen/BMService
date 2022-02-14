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
using namespace bm;
#define OUTPUT_RESULT_FILE  "yolov5_result.json"

std::map<size_t, std::string> globalLabelMap;
std::map<std::string, size_t> globalImageIdMap;
std::map<size_t, size_t> categoryInCoco;
std::map<std::string, std::vector<DetectBox>> globalGroundTruth;

struct YOLOv5Config {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    int devId;
    cv::SophonDevice sophonDev;
    std::vector<bm_image> planar;
    std::vector<bm_image> bm_resizeds;


    // use static to cache resizedImage, to avoid allocating memory everytime
    cv::Size dsize = cv::Size(640, 640);

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
    const size_t classNum = 80;
    float input_scale;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;

        devId = cv::bmcv::getId(ctx->handle);
        sophonDev = cv::SophonDevice(devId);

        ctx->setConfigData(this);
        isNCHW = inTensor->shape(1) == 3;
        input_scale = 1.0;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32; 
            probThreshold = 0.5;
            iouThreshold = 0.5;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            probThreshold = 0.5;
            iouThreshold = 0.5;
            input_scale = inTensor->get_scale();
        }

        if(!isNCHW){
            std::cout<<"is nhwc"<<std::endl;
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED; 
            BM_ASSERT_EQ(inTensor->shape(3), 3);
            grayImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth*3, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, 64);
            bm_device_mem_t planarMem;
            bm_image_get_contiguous_device_mem(planar.size(), planar.data(), &planarMem);
            bm_image_attach_contiguous_mem(grayImages.size(), grayImages.data(), planarMem);
            preOutImages = ctx->allocImagesWithoutMem(
                        netBatch, netHeight, netWidth*3, FORMAT_GRAY, netDtype);
        } else {
            std::cout<<"is nchw"<<std::endl;
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
            netBatch = inTensor->shape(0);
            netFormat = FORMAT_RGB_PLANAR; // for NHWC input
            preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
        }

        planar = ctx->allocImagesWithoutMem(
                            netBatch, netHeight, netWidth, netFormat, DATA_TYPE_EXT_1N_BYTE, 1);
        dsize.height = netHeight;
        dsize.width = netWidth;

    }
};


using InType = std::vector<std::string>;

struct PostOutType {
    std::vector<std::string> rawIns;
    std::vector<std::vector<DetectBox>> results;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

/**
 * @brief   1.
 *          2.
 *          3.
 * 
 * @param in 
 * @param inTensors 
 * @param ctx 
 * @return true 
 * @return false 
 */
bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    if(in.empty()) return false;
    const cv::Scalar color(114, 114, 114);
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    thread_local static YOLOv5Config cfg;
    cfg.initialize(inTensor, ctx);
    const bm_device_mem_t * mem = inTensor->get_device_mem();

    std::vector<cv::Mat> cvImages;
    std::vector<cv::Mat> resizedImages;
    std::vector<std::pair<int, int>> * img0Info = new std::vector<std::pair<int, int>>;

    for(auto imageName: in){
        cv::Mat cvMat = cv::imread(imageName, cv::ImreadModes::IMREAD_COLOR, cfg.devId); 
        cvMat.convertTo(cvMat, CV_32FC3); // uint8 to float32 dtype conversion
        cvImages.push_back(cvMat);
        img0Info->push_back(std::pair<int, int>(cvMat.rows, cvMat.cols));
    }

    /* todo: remove this part */
    std::vector<bm_image> * alignedInputs = new std::vector<bm_image>;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs->push_back(image);
    }  

    aspectScaleAndPad(cfg.sophonDev, cvImages, resizedImages, color, cfg.dsize);
    
    // // to NCHW
    // int8_t *ptr_base = new int8_t[cfg.netHeight * cfg.netWidth * 3 * cfg.netBatch];
    // int8_t *ptr = ptr_base;
    // for (int b = 0; b < cfg.netBatch; b++) {
    //     std::vector<cv::Mat> Channels;
    //     for (int c = 0; c < 3; c++){
    //         cv::Mat channel(cfg.netHeight, cfg.netWidth, CV_8SC1, ptr);
    //         Channels.push_back(channel);
    //         ptr += cfg.netHeight * cfg.netWidth;
    //     }
    //     resizedImages[b] = resizedImages[b] / 255 * cfg.input_scale;
    //     resizedImages[b].convertTo(resizedImages[b], CV_8SC3);
    //     cv::split(resizedImages[b], Channels);
    // }
    void* pInput;
    std::vector<cv::Mat> input_channels; // batch_size * 3
    for (int b = 0; b < cfg.netBatch; b++) {
        resizedImages[b].convertTo(resizedImages[b], CV_8SC3, cfg.input_scale / 255);
    }

    // toNCHW(cfg.sophonDev, resizedImages, pInput, input_channels);
    // bm_memcpy_s2d(ctx->handle, *mem, pInput);

    ctx->setPreExtra(alignedInputs);
    // delete ptr;
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

bool yoloV5BoxParse(DetectBox& box, 
                        float* data, 
                        size_t len, 
                        float probThresh, 
                        CoordConvertInfo& ci,  
                        size_t grid_x,
                        size_t grid_y, 
                        size_t tensor_idx,
                        size_t anchor_idx,
                        size_t grid_w,
                        size_t grid_h,
                        YOLOv5Config cfg){
    size_t classNum = len - 5;
    auto scores = &data[5];
    int category;
    category = argmax(scores, classNum);
    box.confidence = sigmoid(data[4]) * sigmoid(scores[category]);
    if(box.confidence <= probThresh){
        return false;
    }

    float centerX = (sigmoid(data[0]) * 2 - 0.5 + grid_x) * cfg.netWidth / grid_w;
    float centerY = (sigmoid(data[1]) * 2 - 0.5 + grid_y) * cfg.netHeight / grid_h; //center_y
    float width   = pow((sigmoid(data[2]) * 2), 2) * cfg.m_anchors[tensor_idx][anchor_idx][0]; //in w
    float height  = pow((sigmoid(data[3]) * 2), 2) * cfg.m_anchors[tensor_idx][anchor_idx][1]; //in h
    box.xmin  = centerX - width  / 2;
    box.ymin  = centerY - height / 2;
    box.xmax  = centerX + width / 2;
    box.ymax  = centerY + height / 2;

    if(box.xmin > box.xmax || box.ymin > box.ymax){
        return false;
    }

    box.category = categoryInCoco[category];
    box.categoryName = globalLabelMap[category];

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
    size_t anchors = outTensor->shape(1);
    std::vector<size_t> boxNums(outTensors.size());
    size_t totalBoxNum=0;
    size_t classNum = outTensor->shape(4)-5;
    auto singleDataSize = outTensor->shape(4);
    BM_ASSERT_EQ(classNum, cfg.classNum);
    BM_ASSERT_EQ(batch, pInputImages->size());

    std::vector<CoordConvertInfo> coordInfos(batch);
    for(size_t b=0; b<realBatch; b++){
        auto& ci = coordInfos[b];

        ci.inputWidth = (*pInputImages)[b].width;
        ci.inputHeight = (*pInputImages)[b].height;
        ci.ioRatio = std::max((float)ci.inputWidth/cfg.netWidth,
                               (float)ci.inputHeight/cfg.netHeight);
        ci.oiRatio = 1/ci.ioRatio;
        ci.hOffset = (cfg.netHeight - ci.oiRatio * ci.inputHeight)/2;
        ci.wOffset = (cfg.netWidth - ci.oiRatio * ci.inputWidth)/2;
    }

    for(size_t i=0; i<outTensors.size(); i++){
        auto tensor = outTensors[i];
        BM_ASSERT_EQ(batch, tensor->shape(0));
        BM_ASSERT_EQ(classNum, tensor->shape(4)-5);
        boxNums[i] = tensor->partial_shape_count(1,3);
        totalBoxNum += boxNums[i];
    }
    std::vector<std::vector<DetectBox>> batchBoxInfos(batch, std::vector<DetectBox>(totalBoxNum));

    // fill batchBoxInfo
    std::vector<int> batchIndice(batch, 0);
    for(size_t i=0; i<outTensors.size(); i++){
        auto rawData = outTensors[i]->get_float_data();
        size_t grid_w = outTensors[i]->shape(2);
        size_t grid_h = outTensors[i]->shape(3);
        size_t area = grid_w*grid_h;
        auto boxNum = boxNums[i]; // 3*80*80
        
        // when output shape equal to [b, a, h, w, xywh+obj+cls]
        size_t bbox_len = outTensors[i]->shape(4);
        size_t w_stride = bbox_len;
        size_t h_stride = grid_w * w_stride;
        size_t a_stride = grid_h * h_stride;
        size_t b_stride = anchors * a_stride;
        
        for(size_t b=0; b<batch; b++){
            auto& ci = coordInfos[b];
            for(int anchor=0; anchor<anchors; anchor++){         
                for(int grid_y=0; grid_y<grid_h; grid_y++){
                    for(int grid_x=0; grid_x<grid_w; grid_x++){
                        auto rawOffset = anchor * a_stride + grid_y * h_stride + grid_x * w_stride;
                        auto rawBoxData = rawData + rawOffset;
                        auto& boxInfo = batchBoxInfos[b][batchIndice[b]];
                        if(yoloV5BoxParse(boxInfo,
                                        rawBoxData,
                                        singleDataSize, 
                                        cfg.probThreshold, 
                                        ci, 
                                        grid_x, 
                                        grid_y, 
                                        i, 
                                        anchor, 
                                        grid_w, 
                                        grid_h,
                                        cfg)){
                            batchIndice[b]++;
                        }
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
            r.xmin = (r.xmin - ci.wOffset)*ci.ioRatio;
            r.xmax = (r.xmax - ci.wOffset)*ci.ioRatio;
            r.ymin = (r.ymin - ci.hOffset)*ci.ioRatio;
            r.ymax = (r.ymax - ci.hOffset)*ci.ioRatio;
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
            label+= ":" + std::to_string(box.confidence);
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
    // std::string dataPath = topDir + "data/coco/images";
    std::string dataPath = "/workspace/my/dataset/coco/images/oneImg";
    std::string bmodel = topDir + "models/yolov5x/yolov5x_1b_int8.bmodel";
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

