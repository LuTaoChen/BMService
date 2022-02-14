#include <stdio.h>
#include <math.h>
#include<vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cassert>
#include "BMImageUtils.h"
#include "BMCommonUtils.h"
#include "BMLog.h"

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"

namespace bm {

float softmax(float * data, int len){
    float sum = 0;
    for (int i=0;i<len;i++){
        data[i] = exp(data[i]);
        sum += data[i];
    }
    for (int i=0;i<len;i++){
        data[i] /= sum;
    }

}

std::vector<int> calcImageStride(int height, int width,
                                 bm_image_format_ext format,
                                 bm_image_data_format_ext dtype,
                                 int align_bytes){
    int data_size = 1;
    switch (dtype) {
    case DATA_TYPE_EXT_FLOAT32:
        data_size = 4;
        break;
    case DATA_TYPE_EXT_4N_BYTE:
    case DATA_TYPE_EXT_4N_BYTE_SIGNED:
        data_size = 4;
        break;
    default:
        data_size = 1;
        break;
    }

    std::vector<int> stride;
    switch (format) {
    case FORMAT_YUV420P: {
        stride.resize(3);
        stride[0] = width * data_size;
        stride[1] = (FFALIGN(width, 2) >> 1) * data_size;
        stride[2] = stride[1];
        break;
    }
    case FORMAT_YUV422P: {
        stride.resize(3);
        stride[0] = width * data_size;
        stride[1] = (FFALIGN(width, 2) >> 1) * data_size;
        stride[2] = stride[1];
        break;
    }
    case FORMAT_YUV444P: {
        stride.assign(3, FFALIGN(width*data_size, align_bytes));
        break;
    }
    case FORMAT_NV12:
    case FORMAT_NV21: {
        stride.resize(2);
        stride[0] = width * data_size;
        stride[1] = FFALIGN(width, 2) * data_size;
        break;
    }
    case FORMAT_NV16:
    case FORMAT_NV61: {
        stride.resize(2);
        stride[0] = width * data_size;
        stride[1] = FFALIGN(width, 2) * data_size;
        break;
    }
    case FORMAT_GRAY: {
        stride.assign(1, FFALIGN(width * data_size, align_bytes));
        break;
    }
    case FORMAT_COMPRESSED: {
        break;
    }
    case FORMAT_BGR_PACKED:
    case FORMAT_RGB_PACKED: {
        stride.assign(1, FFALIGN(width * 3 * data_size, align_bytes));
        break;
    }
    case FORMAT_BGR_PLANAR:
    case FORMAT_RGB_PLANAR: {
        stride.assign(1, FFALIGN(width * data_size, align_bytes));
        break;
    }
    case FORMAT_BGRP_SEPARATE:
    case FORMAT_RGBP_SEPARATE: {
        stride.resize(3, FFALIGN(width * data_size, align_bytes));
        break;
    }
    default:{

    }
    }
    return stride;
}

bm_image readAlignedImage(bm_handle_t handle, const std::string &name, bm_image_format_ext outFormat){
// TimeRecorder r;
    auto devId = cv::bmcv::getId(handle);
// r.record("read \t device" + std::to_string(devId));
    auto cvImage = cv::imread(name, cv::ImreadModes::IMREAD_COLOR, devId);
// r.record("toBMI");
    bm_image bmImage, alignedImage;
    cv::bmcv::toBMI(cvImage, &bmImage);
    int stride1[3], stride2[3];
    bm_image_get_stride(bmImage, stride1);
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
// r.record("create");
    bm_image_create(handle, bmImage.height, bmImage.width, bmImage.image_format, bmImage.data_type,
                    &alignedImage, stride2);
// r.record("alloc");    
    bm_image_alloc_dev_mem(alignedImage, BMCV_IMAGE_FOR_IN);

    bmcv_copy_to_atrr_t copyToAttr;
    memset(&copyToAttr, 0, sizeof(copyToAttr));
    copyToAttr.start_x = 0;
    copyToAttr.start_y = 0;
    copyToAttr.if_padding = 1;
// r.record("copy");
    bmcv_image_copy_to(handle, copyToAttr, bmImage, alignedImage);
    // for(int i=0; i<1; i++){
    //     dumpImage(alignedImage, "preOut");
    // }

// r.record("destroy");
    bm_image_destroy(bmImage);
// r.show();
    return alignedImage;
}

void centralCropAndResize(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          float centralFactor){
   int numImage = srcImages.size();
   std::vector<bmcv_rect_t> rects(numImage);
   for(int i=0; i<numImage; i++){
       auto& rect = rects[i];
       auto& srcImage = srcImages[i];
       int height = srcImage.height;
       int width = srcImage.width;
       rect.start_x = round(width*(1-centralFactor)*0.5);
       rect.crop_w = round(width*centralFactor);
       rect.start_y = round(height*(1-centralFactor)*0.5);
       rect.crop_h = round(height*centralFactor);
   }
   std::vector<int> cropNumVec(numImage, 1);
   bmcv_image_vpp_basic(handle, numImage, srcImages.data(), dstImages.data(),
                        cropNumVec.data(), rects.data());
    
}

void centralCrop(bm_handle_t handle,
                std::vector<bm_image>& srcImages,
                std::vector<bm_image>& dstImages){
   int numImage = srcImages.size();
   std::vector<bmcv_rect_t> rects(numImage);
   for(int i=0; i<numImage; i++){
       auto& rect = rects[i];
       auto& srcImage = srcImages[i];
       int height = srcImage.height;    
       int width = srcImage.width;
       rect.start_x = width>height?round((width - height) * 0.5):0;
       rect.crop_w = height<width?height:width;
       rect.start_y = height>width?round((height - width) * 0.5):0;
       rect.crop_h = rect.crop_w;
   }
   std::vector<int> cropNumVec(numImage, 1);
   auto statu = bmcv_image_vpp_basic(handle, numImage, srcImages.data(), dstImages.data(),
                        cropNumVec.data(), rects.data());
}

/*
    2021-09-10
    operated by opencv
    dstImages - Discontinuous, no copy
*/
void centralCrop(std::vector<cv::Mat>& srcImages,
                std::vector<cv::Mat>& dstImages){
    int numImage = srcImages.size();
    for(int i=0; i<numImage; i++){
        auto& srcImage = srcImages[i];
        int height = srcImage.size().height;    
        int width = srcImage.size().width;
        int lowerbound = height<width?height:width;
        int L = width>height?round((width - height) * 0.5):0;
        int T = height>width?round((height - width) * 0.5):0;
        dstImages.push_back(srcImage(cv::Rect(L, T, lowerbound, lowerbound)));
    }
}

/*
    2021-08-22
    dstImages: not allocated yet
    The shortest side was resized to lowerbound while keeping the ratio
*/
void aspectRsize(bm_handle_t handle,
                bm_image_format_ext netFormat,
                std::vector<bm_image>& srcImages,
                std::vector<bm_image>& dstImages,
                int lowerBound){    
    int numImage = srcImages.size();
    bm_image_data_format_ext dtype = srcImages.data()[0].data_type;

    int cropH = lowerBound;
    int cropW = lowerBound;

    bmcv_resize_image resizeAttrs[numImage];
    std::vector<bmcv_rect_t> rects(numImage);
    bmcv_resize_t resizeImgAttr[numImage];

    
    for(int i=0; i<numImage; i++){
        auto& srcImage = srcImages[i];
        int height = srcImage.height;
        int width = srcImage.width;
        float ratio;

        // Calculate attribute for each image and crop start point after resize 
        resizeImgAttr[i].start_x = 0;
        resizeImgAttr[i].start_y = 0;
        resizeImgAttr[i].in_width = width;
        resizeImgAttr[i].in_height = height;
        if(height>=width) {
            ratio = float(lowerBound) / float(width);
            resizeImgAttr[i].out_width = cropW;
            resizeImgAttr[i].out_height = ratio * height;
        }
        else{
            ratio = float(lowerBound) / float(height);
            resizeImgAttr[i].out_height = cropH;
            resizeImgAttr[i].out_width = ratio * width;
        }
     
        // Allocate memory for each dst image
        auto stride = calcImageStride(resizeImgAttr[i].out_height + 2, resizeImgAttr[i].out_width + 2, netFormat, dtype, 64);
        bm_image image;
        bm_image_create(handle, resizeImgAttr[i].out_height + 2, resizeImgAttr[i].out_width + 2,
                        netFormat, dtype, &image, stride.data());
        bm_image_alloc_dev_mem(image, BMCV_IMAGE_FOR_IN);
        dstImages.push_back(image);
        auto& dstImage = dstImages[i];

        // bmcv_image_vpp_convert(handle, numImage, srcImages.data()[i], dstImages.data() + i, NULL, BMCV_INTER_LINEAR);
        bmcv_padding_atrr_t padding_attr = {1,
                                            1,
                                            (unsigned int)resizeImgAttr[i].out_width,
                                            (unsigned int)resizeImgAttr[i].out_height,
                                            0,
                                            0,
                                            0,
                                            1};
        bmcv_rect_t crop_rect = {0, 0, srcImage.width, srcImage.height};
        auto ret = bmcv_image_vpp_convert_padding(handle, 1, srcImage, &dstImage, &padding_attr, &crop_rect);
        assert(BM_SUCCESS == ret);
    }
    // std::vector<int> cropNumVec(numImage, 1);
    // bmcv_image_vpp_basic(handle, numImage, srcImages.data(), dstImages.data(), cropNumVec.data(), NULL);
    
}

void dumpImage(bm_image& bmImage, const std::string& name){
    auto fp = fopen(name.c_str(), "w");
    int plane_num = bm_image_get_plane_num(bmImage);
    int* sizes = new int[plane_num];
    auto buffers = new void*[plane_num];
    bm_image_get_byte_size(bmImage, sizes);
    for(int i=0; i<plane_num; i++){
        buffers[i] = new unsigned char[sizes[i]];
    }
    bm_status_t statu = bm_image_copy_device_to_host(bmImage, buffers);

    fprintf(fp, "plane_num=%d\n", plane_num);
    for(int i=0; i<plane_num; i++){
        fprintf(fp, "plane_size=%d\n", sizes[i]);
        for(int j=0; j<sizes[i]; j++){
            if(bmImage.data_type == DATA_TYPE_EXT_1N_BYTE){
                auto data=(unsigned char*) buffers[i];
                fprintf(fp, "%d\n", data[j]);
            } else if(bmImage.data_type == DATA_TYPE_EXT_1N_BYTE_SIGNED){
                auto data=(char*) buffers[i];
                fprintf(fp, "%d\n", data[j]);
            }
        }
        delete [] ((unsigned char*)buffers[i]);
    }
    delete [] buffers;
    delete [] sizes;
    fclose(fp);
}

void saveImage(bm_image& bmImage, const std::string& name){
    cv::Mat cvImage;
    cv::bmcv::toMAT(&bmImage, cvImage);
    cv::imwrite(name, cvImage);
}

static bool split_id_and_label(const std::string& line, size_t& id, std::string& label){
    auto iter = std::find(line.begin(), line.end(), ':');
    if(iter == line.end()){
        id++;
        label = line;
    } else {
        std::string classStr(line.begin(), iter);
        label = std::string(iter+1, line.end());
        id = std::stoul(classStr);
    }
}

std::map<size_t, std::string> loadLabels(const std::string &filename)
{
    BMLOG(INFO, "Loading prediction label file %s", filename.c_str());
    std::ifstream ifs(filename);
    std::string line, label;
    size_t classId = -1;
    std::map<size_t, std::string> labelMap;

    size_t printCount = 0;
    while(std::getline(ifs, line)){
        split_id_and_label(line, classId, label);
        labelMap[classId] = label;
        if(printCount<100){
            BMLOG(INFO, " label #%d: %s", classId, label.c_str());
        } else if(printCount == 100){
            BMLOG(INFO, " ...");
        }
        printCount++;
    }
    BMLOG(INFO, "Loading prediction label file %s done", filename.c_str());
    return labelMap;
}

std::map<std::string, size_t> loadClassRefs(const std::string &filename, const std::string& prefix)
{
    std::ifstream ifs(filename);
    std::string line, label;
    std::map<std::string, size_t> classMap;
    while(std::getline(ifs, line)){
        auto iter = std::find(line.begin(), line.end(), ' ');
        auto name = std::string(line.begin(), iter);
        auto idStr = std::string(iter+1, line.end());
        auto id = std::stol(idStr);
        classMap[prefix+name] = id;
    }
    return classMap;
}

void aspectScaleAndPadSingle(bm_handle_t handle,
                             bm_image& srcImage, bm_image& dstImage, bmcv_color_t color){
    auto srcHeight = srcImage.height;
    auto srcWidth = srcImage.width;
    auto dstHeight = dstImage.height;
    auto dstWidth = dstImage.width;
    bmcv_rect_t cropRect = {0, 0, srcWidth, srcHeight};
    bmcv_padding_atrr_t padAttr;
    padAttr.if_memset = 1;
    padAttr.dst_crop_stx = 0;
    padAttr.dst_crop_sty = 0;
    padAttr.padding_b = color.b;
    padAttr.padding_g = color.g;
    padAttr.padding_r = color.r;
    auto HRatio = (float)dstHeight/srcHeight;
    auto WRatio = (float)dstWidth/srcWidth;
    if(HRatio <= WRatio){
        padAttr.dst_crop_h = dstHeight;
        padAttr.dst_crop_w = srcWidth* HRatio;
        padAttr.dst_crop_stx = (dstWidth - padAttr.dst_crop_w)/2;
    } else {
        padAttr.dst_crop_w = dstWidth;
        padAttr.dst_crop_h = srcHeight * WRatio;
        padAttr.dst_crop_sty = (dstHeight- padAttr.dst_crop_h)/2;
    }
    auto ret = bmcv_image_vpp_convert_padding(handle, 1, srcImage, &dstImage, &padAttr, &cropRect);
    assert(BM_SUCCESS == ret);
}

void aspectScaleAndPad(bm_handle_t handle,
                       std::vector<bm_image> &srcImages,
                       std::vector<bm_image> &dstImages,
                       bmcv_color_t padColor){
        for(size_t i=0; i<srcImages.size(); i++){
            aspectScaleAndPadSingle(handle, srcImages[i], dstImages[i], padColor);
        }
    }

/**
 * @date 2022-01-20
 * @brief using opencv2
 * 
 */
void aspectScaleAndPad(cv::SophonDevice &device,
                        std::vector<cv::Mat>& srcImages,
                        std::vector<cv::Mat>& dstImages,
                        const cv::Scalar & value,
                        const cv::Size & dsize){
    int numImage = srcImages.size();
    int dtype = srcImages[0].type();
    // int dtype = CV_32FC3;
    int dstH = dsize.height;
    int dstW = dsize.width;

    for(int i=0; i<numImage; i++){
        auto& srcImage = srcImages[i];
        int srcH = srcImage.size().height;
        int srcW = srcImage.size().width;
        float ratio;
        cv::Size unpadSize;
        cv::Mat resized = cv::Mat(unpadSize, dtype, device);
        // Calculate attribute for each image and crop start point after resize 
        if(srcH>=srcW) {
            ratio = float(srcH) / float(srcW);
            unpadSize.height = dstH;
            unpadSize.width = dstH / ratio;
        }
        else{
            ratio = float(srcW) / float(srcH);
            unpadSize.width = dstW;
            unpadSize.height = dstW / ratio;
        }
        cv::resize(srcImage, resized, unpadSize, cv::INTER_LINEAR);

        cv::Mat dstImage = cv::Mat(dsize, dtype, device);
        float dw = (float)(dstW - resized.cols)/2; 
        float dh = (float)(dstH - resized.rows)/2;
        int top = round(dh - 0.1);
        int bottom = round(dh + 0.1);
        int left = round(dw - 0.1);
        int right = round(dw + 0.1);
        cv::copyMakeBorder(resized,
                            dstImage, 
                            top,
                            bottom,
                            left, 
                            right,
                            cv::BORDER_CONSTANT, 
                            value);
        dstImages.push_back(dstImage);
    }
}

/**
 * @brief from nhwc to nchw using opencv
 * 
 * @param srcImg all images size and dtype must equal to each other
 * @param continuousData pointer to nchw data, must be continuous
 * @param dstChannels will attached to continuousData
 */
void toNCHW(cv::SophonDevice&       device, 
            std::vector<cv::Mat>&   srcImg, 
            void*                   continuousData, 
            std::vector<cv::Mat>&   dstChannels){
    int w = srcImg[0].cols;
    int h = srcImg[0].rows;
    int dtype = srcImg[0].type();
    for (int i = 0; i < srcImg.size(); i++){
        assert(srcImg[i].cols == w && srcImg[i].rows == h);
        assert(srcImg[i].type() == dtype && srcImg[i].type() == dtype);
        if (dtype % 8 == 0 || dtype % 8 == 1) 
        {
            uchar *channel_base = (uchar *)continuousData + i*3*h*w;
            for (int i = 0; i < 3; i++) {
                cv::Mat channel(h, w, CV_8UC1, channel_base);
                dstChannels.push_back(channel);
                channel_base += h * w;
            }
        } 
        else if(dtype % 8 == 4 || dtype % 8 == 5) 
        {
            float *channel_base = (float *)continuousData + i*3*h*w;
            for (int i = 0; i < 3; i++) {
                cv::Mat channel(h, w, CV_32FC1, channel_base);
                dstChannels.push_back(channel);
                channel_base += h * w;
            }
        }
        else
        {
            std::cout << "ERROR: NOT SUPPORT TYPE!" << std::endl;
            std::exit(0);
        }
        cv::split(srcImg, dstChannels);
        std::cout << "Convert to NCHW complited ! ..." << std::endl;
    }
}
}
