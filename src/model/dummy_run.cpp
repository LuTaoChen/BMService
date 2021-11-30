#include <dirent.h>
#include<vector>
#include<thread>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "bmcv_api.h"
#include <regex>

using namespace bm;
using InType = std::vector<std::string>;    
using ClassId = size_t;

struct PostOutType {
    InType rawIns;
    std::vector<std::vector<std::pair<ClassId, float>>> classAndScores;
};

struct ResNetConfig {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    // use static to cache resizedImage, to avoid allocating memory everytime
    std::vector<bm_image> cropedImages;
    std::vector<bm_image> preOutImages;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }else{
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
            netFormat = FORMAT_RGB_PLANAR; // for NHWC input

        }
        float input_scale = inTensor->get_scale();
        float scale = 1.0;
        float bias = 0;
        float real_scale = scale * input_scale;
        float real_bias = bias * input_scale;
        ConvertAttr.alpha_0 = real_scale;
        ConvertAttr.beta_0 = real_bias - 123.68;
        ConvertAttr.alpha_1 = real_scale;
        ConvertAttr.beta_1 = real_bias - 116.78;
        ConvertAttr.alpha_2 = real_scale;
        ConvertAttr.beta_2 = real_bias - 103.94;

        preOutImages = ctx->allocAlignedImages(netBatch, netHeight, netWidth, netFormat, netDtype);
    }
};
/*
    @param: inTensor: input of model, vector of TensorPtr
*/
bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    thread_local static ResNetConfig cfg;
    if(in.empty()) return false;
    auto inTensor = inTensors[0];
    cfg.initialize(inTensor, ctx);
    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);
    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.rawIns = rawIn;
    auto outTensor = outTensors[0];
    float* data = outTensor->get_float_data();
    std::cout<<*data<<std::endl;
    return true;
}

struct Top5AccuracyStat {
    size_t samples=0;
    size_t top1=0;
    size_t top5=0;
    void show() {
        BMLOG(INFO, "Accuracy: top1=%g%%, top5=%g%%", 100.0*top1/samples, 100.0*top5/samples);
    }
};

bool resultProcess(const PostOutType& out, Top5AccuracyStat& stat){
    if(out.rawIns.empty()) {
        std::cout<<"====="<<std::endl;
        return false;
    }
    return true;
}


int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string datapath = "../data/ILSVRC2012/images";
    // std::string bmodel = topDir + "models/resnet50_v1/fix8b.bmodel";
    std::string bmodel = topDir + "models/resnet101/fix8b_4n.bmodel";
    // std::string bmodel = topDir + "models/resnet50_v1/compilation_4n.bmodel";
    std::string refFile = topDir + "data/ILSVRC2012/val.txt";
    std::string labelFile = topDir + "data/ILSVRC2012/labels.txt";
    if(argc>1) bmodel = argv[1];
    if(argc>2) datapath = argv[2];
    if(argc>3) labelFile = argv[3];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("resnet");
    Top5AccuracyStat topStat;
    std::thread dataThread([datapath, batchSize, &runner](){
        forEachBatch(datapath, batchSize, [&runner](const InType& imageFiles){
            runner.push(imageFiles);
            return true;
        });
        while(!runner.allStopped()){
            if(runner.canPush()) {
                runner.push({});
            } else {
                std::this_thread::yield();
            }
        }
    });
    std::thread resultThread([&runner, &info, batchSize](){
        PostOutType out;
        std::shared_ptr<ProcessStatus> status;
        bool stopped = false;
        Top5AccuracyStat stat;
        while(true){
            while(!runner.pop(out, status)) {
                if(runner.allStopped()) {
                    stopped = true;
                    break;
                }
                std::this_thread::yield();
            }
            if(stopped) break;
            info.update(status, batchSize);

            if(!resultProcess(out, stat)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                // stat.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

