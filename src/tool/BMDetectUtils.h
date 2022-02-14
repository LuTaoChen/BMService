#ifndef BMDETECTUTILS_H
#define BMDETECTUTILS_H

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "bmcv_api.h"

namespace bm {
float sigmoid(float x);
struct DetectBox {
    bool matched; //for eval
    size_t imageId;
    size_t category;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;
    std::string categoryName;
    float iou(const DetectBox& b1);
    bool isValid(float width=1.0, float height=1.0) const;
    bool operator < (const DetectBox& other) const {
        return confidence < other.confidence;
    }
    bool operator > (const DetectBox& other) const {
        return confidence > other.confidence;
    }
    std::string json() const;
};


void drawDetectBox(bm_image& bmImage, const std::vector<DetectBox>& boxes, const std::string& saveName="");

void drawDetectBoxEx(bm_image& bmImage, const std::vector<DetectBox>& boxes, const std::vector<DetectBox>& trueBoxes, const std::string& saveName="");
std::vector<DetectBox> singleNMS(const std::vector<DetectBox>& info,
                                 float iouThresh, size_t topk = 0, bool useSoftNms=false, float sigma=0.3, bool agnosticNMS=false);
std::vector<DetectBox> singleNMS_agnostic(const std::vector<DetectBox>& info,
                                 float iouThresh, size_t topk = 0, bool useSoftNms=false, float sigma=0.3);

std::ostream& operator<<(std::ostream& os, const DetectBox& box);

std::vector<std::vector<DetectBox>> batchNMS(const std::vector<std::vector<DetectBox>>& batchInfo,
                                             float iouThresh, size_t topk=0, bool useSoftNms=false, float sigma=0.3);


template <typename T, typename Pred = std::function<T(const T &)>>
int argmax(const T *data, 
              size_t len, 
              size_t stride = 1,
              Pred pred = [](const T &v){ return v;}){
    int maxIndex = 0;
    for(size_t i = 1; i < len; i++){
      int idx = i * stride;
      if (pred(data[maxIndex * stride]) < pred(data[idx])) {
          maxIndex = i;
      }
    }
    return maxIndex;
}



// std::map<size_t, std::vector<DetectBox> > readCocoDatasetBBox(const std::string &cocoAnnotationFile);
std::map<std::string, std::vector<DetectBox> > readCocoDatasetBBox(const std::string &cocoAnnotationFile);
std::map<std::string, size_t> readCocoDatasetImageIdMap(const std::string &cocoAnnotationFile);
void readCocoDatasetInfo(const std::string &cocoAnnotationFile, std::map<std::string, size_t> &nameToId, std::map<std::string, size_t>& nameToCategory);
void saveCocoResults(const std::vector<DetectBox>& results, const std::string &filename);

}
#endif // BMDETECTUTILS_H

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct __tag_detect_result {
  int class_id;
  bool matched; //for eval
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
}st_detect_result;

struct image_desc {
  std::string fullpath;
  std::string basename;
  int index;
  int width;
  int height;
  cv::Mat mat;
  bm_image image;
  bool is_suspend;
  std::vector<st_detect_result> results;
};


typedef struct __tag_st_calc_ap_info {
  int class_id;
  bool matched; //for eval
  float score;
}st_calc_ap_info;

// mAP for evaluate detection model
// bmiva4/NeuralNetwork/SSD_object/ssd_perf/cpp_cv_bmcv_bmrt/mAP.hpp

class mAP {
public:

  mAP(){};
  ~mAP(){};

  /* 
    sorted_outputs: 
      std::list of image_desc
      length = dataset size
    gt_file:
      json file path
  */
  float calc(std::map<int, std::vector<bm::DetectBox>>& sorted_outputs, const std::string &refFile);

private:
  float cal_iou(const bm::DetectBox& b1, const bm::DetectBox& b2);
  void match(std::vector<bm::DetectBox>& preds, std::vector<bm::DetectBox>& refs);
  float calc_ap(const std::vector<st_calc_ap_info>& preds);
  const int class_num_{80}; // coco的80类
  std::vector<int> fn_;
};




