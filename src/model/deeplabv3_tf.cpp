/**
 * @file deeplabv3_tf.cpp
 * @author Lu Taochen 
 * @brief suitable for deeplabv3 in tensorflow model zoo
 *        https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
 * @date 2022-02-09
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
    cv::SophonDevice sophonDev;

    std::vector<bm_image> resizedImages;
    std::vector<bm_image> tempImg;
    std::vector<bm_image> bm_resizeds;

    std::vector<bm_image> preOutImages;
    std::string savePath = "deeplabv3_out";
    float probThreshold;
    float iouThreshold;
    const int classNum = 21;
    int devId;
    cv::Size dsize = cv::Size(512, 512);

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
        if (inTensor->get_dtype() == BM_UINT8) {
            netDtype = DATA_TYPE_EXT_1N_BYTE;
            input_scale = 1.0;
        } else if (inTensor->get_dtype() == BM_INT8) {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            input_scale = inTensor->get_scale();
        } else {
            std::cout << "Unsupport dtype! ..." << std::endl;
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_BGR_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        } else {
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
            netFormat = FORMAT_BGR_PLANAR;
        }
        preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
        dsize.height = netHeight;
        dsize.width = netWidth;
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
    std::vector<cv::Mat> results;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

/**
 * @brief No need to resize, crop, normalize. Need to convert to rgb ...
 * 
 */
bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    thread_local static DeepLabv3Config cfg;
    cfg.initialize(inTensor, ctx);

    std::vector<cv::Mat> cvImages;
    std::vector<cv::Mat> resizedImages;
    std::vector<std::pair<int, int>> * img0Info = new std::vector<std::pair<int, int>>;
    for(auto imageName: in){
        cv::Mat u8Mat = cv::imread(imageName, cv::ImreadModes::IMREAD_COLOR, cfg.devId); 
        cvImages.push_back(u8Mat);
        img0Info->push_back(std::pair<int, int>(u8Mat.rows, u8Mat.cols));
    }

    if (cfg.aspctPreserving) {
        const cv::Scalar color(114, 114, 114);
        aspectScaleAndPad(cfg.sophonDev, cvImages, resizedImages, color, cfg.dsize);
    } else {
        for (cv::Mat img: cvImages) {
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(cfg.netHeight, cfg.netWidth));
            resizedImages.push_back(resized);
        }
    }
    
    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

    for (int i = 0; i < cfg.netBatch; i++){
        bm_image bm_resized;
        cv::cvtColor(resizedImages[i], resizedImages[i], cv::COLOR_BGR2RGB);
        cv::bmcv::toBMI(resizedImages[i], &bm_resized); 
        cfg.bm_resizeds.push_back(bm_resized);
    }

    bmcv_copy_to_atrr_s copy_atrr;
    copy_atrr.start_x = 0;
    copy_atrr.start_y = 0;
    copy_atrr.if_padding = 0;
    for (int i = 0; i < cfg.netBatch; i++){
        bmcv_image_copy_to(ctx->handle, copy_atrr, cfg.bm_resizeds[i], cfg.preOutImages[i]);
    }

    // pass input info to post process
    ctx->setPreExtra(img0Info);

    for(int i=0; i<cfg.netBatch; i++){  
        bm_image_destroy(cfg.bm_resizeds[i]);
    }
    cfg.bm_resizeds.clear();
    return true;
}

/**
 * @brief No argmax, pixel-wise classification
 * 
 * @param postOut call postOut.make_array() in this function
 */
bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){

    postOut.rawIns = rawIn;
    if(rawIn.empty()) return false;
    auto pCfg = (DeepLabv3Config*)ctx->getConfigData();
    auto& cfg = *pCfg;
    auto pImg0Info = reinterpret_cast<std::vector<std::pair<int, int>>*>(ctx->getPostExtra());

    auto realBatch = rawIn.size(); // number of dataset may not devided by batchsize 
    auto outTensor = outTensors[0];
    const bm_shape_t * out_shape = outTensor->get_shape();
    int batch = outTensor->shape(0);
    BM_ASSERT_EQ(batch, cfg.netBatch);
    int output_h = out_shape->dims[1];
    int output_w = out_shape->dims[2];
    int w_stride = 1;
    int h_stride = output_w * w_stride;
    int b_stride = output_h * h_stride;
    uchar* rawData = outTensor->get_raw_data();
    
    std::vector<int> crop_h(realBatch, output_h);
    std::vector<int> crop_w(realBatch, output_w);
    std::vector<int> offset_x(realBatch, 0);
    std::vector<int> offset_y(realBatch, 0);

    if (cfg.aspctPreserving){
        for (int b = 0; b < realBatch; b++) { 
            int h0 = (*pImg0Info)[b].first;
            int w0 = (*pImg0Info)[b].second;
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
    }

    postOut.make_array(realBatch, crop_h, crop_w);
    auto cls_per_pix = postOut.cls_per_pix;

    // fill prebuild array
    for (int b = 0; b < realBatch; b++) {
        for (int i = 0; i < crop_h[b]; i++) {
            for (int j = 0; j < crop_w[b]; j++) {
                int rawOffset =  b * b_stride + (i + offset_y[b]) * h_stride + (j + offset_x[b]);
                int32_t* cls = (int32_t *)rawData + rawOffset;
                int category = (int)(* cls);
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
            cv::resize(result, result, cv::Size((*pImg0Info)[b].second, (*pImg0Info)[b].first),
                            0, 0, cv::INTER_NEAREST);
            cv::imwrite(cfg.savePath + "/" + name, result);
            postOut.results.push_back(result);
        }
    }

    // clear extra data
    delete pImg0Info;
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
    for(auto b = 0; b < batch; b++){
        auto name = baseName(out.rawIns[b]);
        name.replace(11, 4, ".png");

        cv::Mat label_img = cv::imread(segClass+ '/' + name);
        cv::Mat label;
        // cv::resize(label_img, label, cv::Size(out.w[b], out.h[b]), 0, 0, cv::INTER_NEAREST);
        auto conf_mat_single = cm.update(label_img, out.results[b]);

        cm.cal_TPTNFPFN(conf_mat_single);
        std::vector<float> miou = cm.cal_mIoU(conf_mat_single);
        BMLOG(INFO, "%s : mIoU: %f, invalid pixle: %d", name.c_str(), miou.at(miou.size() - 1), cm.get_white());
        for(int i = 0; i < miou.size() - 1; i++){
            if (miou[i] != 0) {
                BMLOG(INFO, "class %d IoU: %f", i, miou[i]);
            }
        }
    }
    out.results.clear();
    out.release();
    return true;
}

int main(int argc, char* argv[]){
    set_env_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/VOC2012/images";
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
    ConfusionMatrix cm(21, voc_palette);
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
        for(int i = 0; i < miou.size()-1; i++){
            if (miou[i] != 0) {
                BMLOG(INFO, "class %d mIoU: %f", i, miou[i]);
            }
        }
    });
    dataThread.join();
    resultThread.join();
    return 0;
}

