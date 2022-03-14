#ifndef BMIMAGEUTILS_H
#define BMIMAGEUTILS_H
#include<string>
#include<map>
#include "bmcv_api.h"

//#define FFALIGN(x, n) ((((x)+((n)-1))/(n))*(n))

namespace bm {

float softmax(float * data, int len);

std::vector<int> calcImageStride(
        int height, int width,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        int align_bytes = 1);

bm_image readAlignedImage(bm_handle_t handle, const std::string& name,  bm_image_format_ext outFormat=FORMAT_RGB_PACKED);
bm_image readAlignedImage4N(bm_handle_t handle, const std::string& name,  bm_image_format_ext outFormat=FORMAT_RGB_PACKED);


// for inceptionv3
void centralCropAndResize(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          float centralFactor = 0.875);


// default resnet and vgg in tensorflow slim lib
void aspectRsize(bm_handle_t handle,
                bm_image_format_ext netFormat,
                std::vector<bm_image>& srcImages,
                std::vector<bm_image>& dstImages,
                int lowerBound = 224);
void aspectRsize(cv::SophonDevice &device,
                std::vector<cv::Mat>& srcImages, 
                std::vector<cv::Mat>& dstImages, 
                int lowerBound = 224);


void centralCrop(bm_handle_t handle,
                std::vector<bm_image>& srcImages,
                std::vector<bm_image>& dstImages);

void centralCrop(std::vector<cv::Mat>& srcImages, std::vector<cv::Mat>& dstImages);


void aspectScaleAndPad(bm_handle_t handle,
                        std::vector<bm_image>& srcImages,
                        std::vector<bm_image>& dstImages,
                        bmcv_color_t padColor);

void aspectScaleAndPad(cv::SophonDevice &device,
                        std::vector<cv::Mat>& srcImages,
                        std::vector<cv::Mat>& dstImages,
                        const cv::Scalar & value = cv::Scalar(114, 114, 114),
                        const cv::Size & dsize = cv::Size(640, 640));

void saveImage(bm_image& bmImage, const std::string& name = "image.jpg");
void dumpImage(bm_image& bmImage, const std::string& name = "image.txt");

std::map<size_t, std::string> loadLabels(const std::string& filename);
std::map<std::string, size_t> loadClassRefs(const std::string& filename, const std::string& prefix="");

/**
 * @brief from nhwc to nchw using opencv
 * 
 * @param srcImg all images size and dtype must equal to each other
 * @param continuousData pointer to nchw data, must be continuous
 * @param dstChannels will attached to continuousData
 */
void toNCHW(cv::SophonDevice &device, std::vector<cv::Mat>& srcImg, void *continuousData, std::vector<std::vector<cv::Mat>>& dstChannels);

}




#endif // BMIMAGEUTILS_H
