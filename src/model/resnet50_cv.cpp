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
    std::vector<cv::Mat> cropedImages;
    std::vector<cv::Mat> cvImages;
    std::vector<cv::Mat> aspectResized;

    // used as a wrapper of input tensor
    std::vector<bm_image> preOutImages;
    cv::Mat BGRMean;
    int devId;
    cv::SophonDevice sophonDev;
    float input_scale;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) 
            return;
        initialized = true;
        devId = cv::bmcv::getId(ctx->handle);
        sophonDev = cv::SophonDevice(devId);
        input_scale = inTensor->get_scale();
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
            netFormat = FORMAT_RGB_PLANAR; 
        }

        BGRMean = cv::Mat(netWidth, netHeight, CV_32FC3, cv::Scalar(103.94, 116.78, 123.68), sophonDev);

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

        netDtype = inTensor->get_dtype() == BM_FLOAT32 ? DATA_TYPE_EXT_FLOAT32 : DATA_TYPE_EXT_1N_BYTE_SIGNED;
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED; 
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }
        preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
    }
};

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    /*
    @param: inTensor: input of model, vector of TensorPtr
    */
    thread_local static ResNetConfig cfg;
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    cfg.initialize(inTensor, ctx);
    // after aspect preserve

    // std::regex r("ILSVRC2012_val_\\d+\\.JPEG");
    // std::smatch m; 
    // bool found = regex_search(in[0], m, r);

// TimeRecorder r;
// r.record("attach memery");  
    // attach preprocess output to bmodel input
    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

// r.record("read");
    for(auto imageName: in){
        auto u8Mat = cv::imread(imageName, cv::ImreadModes::IMREAD_COLOR, cfg.devId); 
        cfg.cvImages.push_back(u8Mat);
    }

// r.record("resize and crop");
    centralCrop(cfg.cvImages, cfg.cropedImages);
    aspectScaleAndPad(cfg.sophonDev, cfg.cropedImages, cfg.aspectResized);

    // copy function debug
    /*std::vector<bm_image> bm_imgs1, bm_imgs2;
    for (int i=0; i<cfg.netBatch; i++){
        bm_image bm_img1, bm_img2;
        // cv::Mat B;
        // oriImages[i].convertTo(B, CV_8UC3); //convert to byte dtype
        // B = cfg.cropedImages[i].clone();
        cv::bmcv::toBMI(cfg.cvImages[i], &bm_img1, true);
        // bm_cropedImgs.push_back(bm_cropedImg);
        
        // create bm_image
        int stride1, stride2;
        bm_image_get_stride(bm_img1, &stride1);
        stride2 = FFALIGN(stride1, 64);
        bm_image_create(ctx->handle, bm_img1.height, bm_img1.width, bm_img1.image_format, bm_img1.data_type,
                &bm_img2, &stride2);
        bm_image_alloc_dev_mem(bm_img2, BMCV_IMAGE_FOR_IN);

        // try to copy
        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));
        copyToAttr.start_x = 0;
        copyToAttr.start_y = 0;
        copyToAttr.if_padding = 1;
        bm_status_t statu1 = bmcv_image_copy_to(ctx->handle, copyToAttr, bm_img1, bm_img2);

        // copy to host to debug
        int* size = new int;

        bm_image_get_byte_size(bm_img1, size);
        auto buffers1 = new void*[1];
        buffers1[0] = new unsigned char[*size];
        bm_status_t statu2 = bm_image_copy_device_to_host(bm_img1, buffers1);
            
        bm_image_get_byte_size(bm_img2, size);
        auto buffers2 = new void*[1];
        buffers2[0] = new unsigned char[*size];
        bm_status_t statu3 = bm_image_copy_device_to_host(bm_img2, buffers2);
        
        delete [] buffers1;
        delete [] buffers2;
        delete [] size;
        // bm_imgs2.push_back(bm_img2);
        // bm_image_destroy(bm_img1);
    }
*/
    
    std::vector<bm_image> bm_linears;
    for (int i=0; i<cfg.netBatch; i++){
        bm_image bm_linear;
        if(cfg.netDtype == DATA_TYPE_EXT_FLOAT32){
            cfg.cropedImages[i].convertTo(cfg.cropedImages[i], CV_32FC3); // uint8 to float32 dtype conversion
            cv::Mat linear(cfg.sophonDev);
            linear = cfg.cropedImages[i] - cfg.BGRMean;
            cv::bmcv::toBMI(linear, &bm_linear);
        }
        else if(cfg.netDtype == DATA_TYPE_EXT_1N_BYTE_SIGNED){

            // cv::Mat rgbImg(cfg.netWidth, cfg.netHeight, CV_32FC1, cfg.sophonDev);
            // if(cfg.netDtype == DATA_TYPE_EXT_FLOAT32){
            //     BGRToRGB_opencv<cv::Vec3f>(linear, rgbImg);
            // }else if (cfg.netDtype == DATA_TYPE_EXT_1N_BYTE_SIGNED){
            //     BGRToRGB_opencv<cv::Vec3b>(linear, rgbImg);
            // }

            cv::bmcv::toBMI(cfg.cropedImages[i], &bm_linear); 
            int stride1[3];
            bm_image_get_stride(bm_linear, stride1);
            std::cout<<stride1[0]<<std::endl;
        }
        bm_linears.push_back(bm_linear);
    }
    
    if (cfg.netDtype == DATA_TYPE_EXT_FLOAT32){
        bm_status_t statu = bmcv_image_storage_convert(ctx->handle, 1, bm_linears.data(), cfg.preOutImages.data());
    }else if(cfg.netDtype == DATA_TYPE_EXT_1N_BYTE_SIGNED){
        std::vector<bm_image> rgb_planar;
        rgb_planar = ctx->allocImagesWithoutMem(
                            cfg.netBatch, cfg.netHeight, cfg.netWidth, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, 1);
        bmcv_image_storage_convert(ctx->handle, 4, bm_linears.data(), rgb_planar.data());   // BGR 2 RGB /packed to plannar
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, rgb_planar.data(), cfg.preOutImages.data()); // u8 to s8
    }

    for(int i=0; i<cfg.netBatch; i++){  
        // copy to host to debug
        /*int* size = new int;

        bm_image_get_byte_size(bm_linears[i], size);
        auto buf1 = new void*[1];
        buf1[0] = new unsigned char[*size];
        bm_image_copy_device_to_host(bm_linears[i], buf1);

        bm_image_get_byte_size(cfg.preOutImages[i], size);
        auto buf2 = new void*[1];
        buf2[0] = new unsigned char[*size];
        bm_image_copy_device_to_host(cfg.preOutImages[i], buf2);

        delete [] buf1;
        delete [] buf2;   
        delete [] size;
        */

        cfg.cvImages.pop_back();
        cfg.aspectResized.pop_back();
        cfg.cropedImages.pop_back();
    }
    
    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    const size_t K=5;
    postOut.rawIns = rawIn;
    auto outTensor = outTensors[0];
    float* data = outTensor->get_float_data();
    size_t batch = outTensor->shape(0);
    size_t len = outTensor->shape(1);

    postOut.classAndScores.resize(batch);
    for(size_t b=0; b<batch; b++){
        float* allScores = data+b*len;
        postOut.classAndScores[b] = topk(allScores, len, K);
    }
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

bool resultProcess(const PostOutType& out, Top5AccuracyStat& stat,
                   std::map<std::string, size_t>& refMap,
                   std::map<size_t, std::string>& labelMap){
    if(out.rawIns.empty()) return false;
    BM_ASSERT_EQ(out.rawIns.size(), out.classAndScores.size());
    for(size_t i=0; i<out.rawIns.size(); i++){
        auto& inName = out.rawIns[i];
        auto realClass = refMap[out.rawIns[i]];
        auto& classAndScores = out.classAndScores[i];
        auto firstClass = classAndScores[0].first;
        auto firstScore = classAndScores[0].second;
        stat.samples++;
        stat.top1 += firstClass == realClass;
        for(auto& cs: classAndScores){
            if(cs.first == realClass){
                stat.top5++;
                break;
            }
        }
        BMLOG(INFO, "%s: infer_class=%d: score=%f: real_class=%d: label=%s",
              out.rawIns[i].c_str(), firstClass, firstScore,
              realClass, labelMap[realClass].c_str());
    }
    return true;
}


int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/ILSVRC2012/images";
    // std::string bmodel = topDir + "models/resnet50_v1/fix8b.bmodel";
    // std::string bmodel = topDir + "models/resnet50_v1/compilation.bmodel";
    std::string bmodel = topDir + "models/resnet50_v1/fp32.bmodel";
    std::string refFile = topDir + "data/ILSVRC2012/val.txt";
    std::string labelFile = topDir + "data/ILSVRC2012/labels.txt";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    std::string prefix = dataPath;
    if(prefix[prefix.size()-1] != '/'){
        prefix += "/";
    }
    auto refMap = loadClassRefs(refFile, prefix);
    auto labelMap = loadLabels(labelFile);
    ProcessStatInfo info("resnet50");
    Top5AccuracyStat topStat;
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
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
    std::thread resultThread([&runner, &refMap, &labelMap, &info, batchSize](){
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

            if(!resultProcess(out, stat, refMap, labelMap)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                stat.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}


/*
    2021-09-23
    BGR to RGB and 
    packed to plannar
    convert to fp32/int16
    multiplied by input scale of quantization
    subtracted by mean rgb value btw
    rowType: cv::Vec3f(float), cv::Vec3b(uint8)
    dstMat: width, height*3, CV_8UC1
*/
void fast_preprocess(cv::Mat& srcMat, cv::Mat& dstMat) {
    int height = srcMat.size().height;
    int width = srcMat.size().width;   
    int numerator = 217;
    int c_stride = height*width;
    for (int i = 0; i < srcMat.rows; ++i) {
		// pixel in ith row pointer
		cv::Vec3b *p1 = srcMat.ptr<cv::Vec3b>(i); 
		cv::Vec3b *pr = dstMat.ptr<cv::Vec3b>(i);
        cv::Vec3b *pg = dstMat.ptr<cv::Vec3b>(i + height);
        cv::Vec3b *pb = dstMat.ptr<cv::Vec3b>(i + height*2);
		for(int j=0; j<srcMat.cols; ++j) { 
			// exchange
            short temp;
			temp = (pr[j][2] * numerator)>>8 - 104;
            pr[j] = (int8_t)temp; 
			temp = (pg[j][1] * numerator)>>8 - 117;
            pg[j] = (int8_t)temp; 
			temp = (pb[j][0] * numerator)>>8 - 124;
            pb[j] = (int8_t)temp;
		}
	} 
}

