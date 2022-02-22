/**
 * @file deeplabv3.cpp
 * @author Lu Taochen 
 * @brief suitable for deeplabv3 in mmsegmentation, https://github.com/open-mmlab/mmsegmentation.git
 * @date 2022-01-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <dirent.h>
#include <vector>
#include <map>
#include <thread>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "SegmentationUtils.h"
#include "bmcv_api.h"

using namespace bm;

cv::Vec3b voc_palette[21] = {  
                            cv::Vec3b(0  , 0  , 0  ), 
                            cv::Vec3b(0  , 0  , 128), 
                            cv::Vec3b(0  , 128, 0  ),
                            cv::Vec3b(0  , 128, 128),
                            cv::Vec3b(128, 0  , 0  ),
                            cv::Vec3b(128, 0  , 128),
                            cv::Vec3b(128, 128, 0), 
                            cv::Vec3b(128, 128, 128), 
                            cv::Vec3b(0  , 0  , 64 ),
                            cv::Vec3b(0  , 0  , 192), 
                            cv::Vec3b(0  , 128, 64 ), 
                            cv::Vec3b(0  , 128, 192), 
                            cv::Vec3b(128, 0  , 64 ),
                            cv::Vec3b(128, 0  , 192), 
                            cv::Vec3b(128, 128, 64 ), 
                            cv::Vec3b(128, 128, 192), 
                            cv::Vec3b(0  , 64 , 0  ),
                            cv::Vec3b(0  , 64 , 128), 
                            cv::Vec3b(0  , 192, 0  ), 
                            cv::Vec3b(0  , 192, 128), 
                            cv::Vec3b(128, 64 , 0  )
                        };

struct DeepLabv3Config {
    bool initialized = false;
    bool isNCHW;
    bool save_result;
    bool aspctPreserving = 1;

    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    cv::SophonDevice sophonDev;

    std::vector<bm_image> resizedImages;
    std::vector<bm_image> tempImg;
    std::vector<bm_image> grayImages;
    std::vector<bm_image> preOutImages;
    std::string savePath = "deeplabv3_out";
    float probThreshold;
    float iouThreshold;
    const int classNum = 21;
    int devId;


    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        save_result = true;

        std::string command;
        command = "mkdir -p " + savePath;  
        system(command.c_str());

        devId = cv::bmcv::getId(ctx->handle);
        sophonDev = cv::SophonDevice(devId);
        ctx->setConfigData(this);
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
        float input_scale;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32; 
            input_scale = 1.0;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            input_scale = inTensor->get_scale();
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        } else {
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
            netFormat = FORMAT_RGB_PLANAR;
        }
        ConvertAttr.alpha_0 = 1/58.395 * input_scale;
        ConvertAttr.beta_0 = -123.68/58.395 * input_scale;
        ConvertAttr.alpha_1 = 1/57.12 * input_scale;
        ConvertAttr.beta_1 = -116.78/57.12 * input_scale;
        ConvertAttr.alpha_2 = 1/57.375 * input_scale;
        ConvertAttr.beta_2 = -103.94/57.375 * input_scale;

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

/**
 * @brief To allocate & release a memory for pixel-wise prediction
 *           
 */
struct PostOutType {
    void make_array(int n, std::vector<int>& h, std::vector<int>& w){
        this->n = n;
        for(int b = 0; b < n; b++) {
            area.push_back(h[b] * w[b]);
            this->h.push_back(h[b]);
            this->w.push_back(w[b]);
            uchar *p = new uchar[area[b]];
            cls_per_pix.push_back(p);
        }
    }
    void release(){
        for (auto &p : cls_per_pix) {
            delete p;
        } 
    }
    std::vector<std::string> rawIns;
    std::vector<uchar *> cls_per_pix; // as long as number of classes less equal than 256 otherwise use int
    int n; 
    std::vector<int> h;
    std::vector<int> w;
    std::vector<int> area;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

/**
 * @brief Resize, keep_ratio 
 * Normalize, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], 
 * BGR to RGB
 */
bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    thread_local static DeepLabv3Config cfg;
    cfg.initialize(inTensor, ctx);

    std::vector<bm_image> * alignedInputs = new std::vector<bm_image>;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs->push_back(image);
    }

    bmcv_color_t color = {114, 114, 114};
    aspectScaleAndPad(ctx->handle, *alignedInputs, cfg.resizedImages, color);

    // saveImage(cfg.resizedImages[0], "resize.jpg");
    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);
    bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.resizedImages.data(), cfg.preOutImages.data());
    ctx->setPreExtra(alignedInputs);
    // delete alignedInputs;
    return true;
}

/**
 * @brief Do argmax, pixel-wise classification
 * 
 * @param postOut call postOut.make_array() in this function
 */
bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){

    postOut.rawIns = rawIn;
    if(rawIn.empty()) return false;
    auto pCfg = (DeepLabv3Config*)ctx->getConfigData();
    auto& cfg = *pCfg;
    auto pInputImages = reinterpret_cast<std::vector<bm_image>*>(ctx->getPostExtra());

    auto realBatch = rawIn.size(); // number of dataset may not devided by batchsize 
    auto outTensor = outTensors[0];
    const bm_shape_t * out_shape = outTensor->get_shape();
    int batch = outTensor->shape(0);
    BM_ASSERT_EQ(batch, cfg.netBatch);
    BM_ASSERT_EQ(out_shape->dims[1], cfg.classNum);
    int output_h = out_shape->dims[2];
    int output_w = out_shape->dims[3];
    int c_stride = output_h * output_w;
    int b_stride = cfg.classNum * c_stride;
    float* rawData_fp = outTensors[0]->get_float_data();
        
    std::vector<int> crop_h(realBatch, output_h);
    std::vector<int> crop_w(realBatch, output_w);
    std::vector<int> offset_x(realBatch, 0);
    std::vector<int> offset_y(realBatch, 0);

    for (int b = 0; b < realBatch; b++) { 
        int h0 = (*pInputImages)[b].height;
        int w0 = (*pInputImages)[b].width;
        int cropW = output_w;
        int cropH = output_h;
        if (h0 < w0){
            cropH *= ((float)h0 / (float)w0);
            crop_h[b] = cropH;
            crop_w[b] = cropW;
            offset_y[b] = (output_h - cropH) / 2;
            offset_x[b] = 0;
        } else {
            cropW *= ((float)w0 / (float)h0);
            crop_w[b] = cropW;
            crop_h[b] = cropH;
            offset_x[b] = (output_w - cropW) / 2;
            offset_y[b] = 0;
        }
    }

    postOut.make_array(realBatch, crop_h, crop_w);
    auto cls_per_pix = postOut.cls_per_pix;

    // fill prebuild category array
    for (int b = 0; b < realBatch; b++) {
        for (int i = 0; i < crop_h[b]; i++) {
            for (int j = 0; j < crop_w[b]; j++) {
                int rawOffset = b * b_stride + (i + offset_y[b]) * output_w + (j + offset_x[b]);
                float* cls0 = rawData_fp + rawOffset;
                int category;
                category = argmax(cls0, cfg.classNum, c_stride);
                cls_per_pix[b][i * crop_w[b] + j] = (uchar)category;
            }
        }
    }


    // visulization
    if (cfg.save_result) {
        for(int b = 0; b < realBatch; b++){
            auto name = baseName(rawIn[b]);
            cv::Mat result(crop_h[b], crop_w[b], CV_8UC3);
            SegmentationVisualization(result, cls_per_pix[b], voc_palette);   
            cv::imwrite(cfg.savePath + "/" + name, result);
        }
    }

    // clear extra data
    for(int i=0; i<pInputImages->size(); i++) {
        bm_image_destroy(pInputImages->at(i));
    }
    delete pInputImages;
    // delete[] cls_per_pix;
    return true;
}


/**
 * @brief Build confusion matrix and calculate mIoU if needed.
 *  resize annotation images using nearest algorithm is necessary.
 * 
 * @param out Have to be released in this function
 */
bool resultProcess(PostOutType& out, ConfusionMatrix& cm, std::string& segClass, cv::Vec3b *palette){
    if(out.rawIns.empty()) return false;
    auto batch = out.rawIns.size();
    for(auto b=0; b < batch; b++){
        auto name = baseName(out.rawIns[b]);
        name.replace(11, 4, ".png");

        cv::Mat label_img = cv::imread(segClass+ '/' + name);
        cv::Mat label;
        cv::resize(label_img, label, cv::Size(out.w[b], out.h[b]), 0, 0, cv::INTER_NEAREST);
        auto conf_mat_single = cm.update(label, out.cls_per_pix[b]);

        cm.cal_TPTNFPFN(conf_mat_single);
        std::vector<float> miou = cm.cal_mIoU(conf_mat_single);
        BMLOG(INFO, "%s : mIoU: %f, invalid pixle: %d", name.c_str(), miou.at(miou.size() - 1), cm.get_white());
        for(int i = 0; i < miou.size() - 1; i++){
            if (miou[i] != 0) {
                BMLOG(INFO, "%s : class %d mIoU: %f", name.c_str(), i, miou[i]);
            }
        }
    }
    out.release();
    return true;
}

int main(int argc, char* argv[]){
    set_env_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/VOC2012/images/";
    std::string bmodel = topDir + "models/deeplabv3/compilation.bmodel";
    std::string segClass = topDir + "data/VOC2012/SegmentationClass";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) segClass = argv[3];

    // get info first
    bm_handle_t bm_handle;
    const char **net_names;
    bm_dev_request(&bm_handle, 0);
    void* p_bmrt = bmrt_create(bm_handle);
    bool flag = bmrt_load_bmodel(p_bmrt, bmodel.c_str());
    bmrt_get_network_names(p_bmrt, &net_names);
    auto net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
    const bm_shape_t* out_shape = net_info->stages[0].output_shapes;
    
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    int batchSize= runner.getBatchSize();
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
    ConfusionMatrix cm(out_shape->dims[1], voc_palette);
    std::thread resultThread([&runner, &info, &out_shape, &cm, &segClass](){
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
            if(!resultProcess(out, cm, segClass, voc_palette)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                break;
            }
        }

        // calculate final mIoU
        auto conf_mat = cm.get_confusion_mat();
        std::vector<float> miou = cm.cal_mIoU(conf_mat);
        BMLOG(INFO, "All mIoU: %f ", miou.at(miou.size() - 1));
        for(int i = 0; i < miou.size() - 1; i++){
            if (miou[i] != 0) {
                BMLOG(INFO, "class %d mIoU: %f", i, miou[i]);
            }
        }
    });
    dataThread.join();
    resultThread.join();
    return 0;
}

