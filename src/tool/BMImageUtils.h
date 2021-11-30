#ifndef BMIMAGEUTILS_H
#define BMIMAGEUTILS_H
#include<string>
#include<map>
#include "bmcv_api.h"

//#define FFALIGN(x, n) ((((x)+((n)-1))/(n))*(n))

namespace bm {
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

// for vgg and resnet and so on
void centralCrop(bm_handle_t handle,
                std::vector<bm_image>& srcImages,
                std::vector<bm_image>& dstImages);

void centralCrop(std::vector<cv::Mat>& srcImages, std::vector<cv::Mat>& dstImages);

// for yolov
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
}

/*
    2021-09-17
    OPENCV BGR to RGB
    rowType: cv::Vec3f(float), cv::Vec3b(uint8)
*/
template<class rowType>
void BGRToRGB_opencv(cv::Mat& srcMat, cv::Mat& dstMat) 
{
        for (int i = 0; i < srcMat.rows; ++i) 
        { 
		// pixel in ith row pointer
		rowType *p1 = srcMat.ptr<rowType>(i); 
		rowType *p2 = dstMat.ptr<rowType>(i);
		for(int j=0; j<srcMat.cols; ++j) 
		{ 
			// exchange
			p2[j][2] = p1[j][0]; 
			p2[j][1] = p1[j][1]; 
			p2[j][0] = p1[j][2]; 
		}
	} 
}



#endif // BMIMAGEUTILS_H
