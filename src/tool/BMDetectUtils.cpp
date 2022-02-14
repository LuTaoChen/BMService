#include <fstream>
#include "jsonxx.h"
#include "opencv2/opencv.hpp"
#include "BMCommonUtils.h"
#include "BMDetectUtils.h"
#include "BMLog.h"

namespace  bm {

float sigmoid(float x){
    return 1.0 / (1 + expf(-x));
}

float DetectBox::iou(const DetectBox &b1) {
    auto o_xmin = std::max(xmin, b1.xmin);
    auto o_xmax = std::min(xmax, b1.xmax);
    auto o_ymin = std::max(ymin, b1.ymin);
    auto o_ymax = std::min(ymax, b1.ymax);
    if(o_xmin > o_xmax || o_ymin > o_ymax) {
        return 0;
    }
    auto b0_area = (xmax - xmin) * (ymax - ymin);
    auto b1_area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin);
    auto o_area = (o_xmax - o_xmin) * (o_ymax - o_ymin);
    return (float)o_area / (b0_area + b1_area - o_area);
}

bool DetectBox::isValid(float width, float height) const
{
    return !(xmin >= xmax ||
            ymin >= ymax ||
            xmin<0 || xmax >= width ||
             ymin<0 || ymax >= height);
}

std::string DetectBox::json() const
{
    std::string s = "{";
    s+= "\"image_id\":" + std::to_string(imageId) +",";
    s+= "\"category_id\":" + std::to_string(category) +",";
    s+= "\"score\":" + std::to_string(confidence) +",";
    s+= "\"bbox\": ["
            + std::to_string(xmin) +","
            + std::to_string(ymin) +","
            + std::to_string(xmax - xmin) +","
            + std::to_string(ymax - ymin) +"]}";
    return s;
}

std::vector<std::vector<DetectBox> > batchNMS(const std::vector<std::vector<DetectBox> > &batchInfo, float iouThresh, size_t topk, bool useSoftNms, float sigma){
    std::vector<std::vector<DetectBox>> results(batchInfo.size());
    for(size_t i=0; i<batchInfo.size(); i++){
        results[i] = singleNMS(batchInfo[i], iouThresh, topk, useSoftNms, sigma);
    }
    return results;
}

std::vector<DetectBox> singleNMS(const std::vector<DetectBox> &info, float iouThresh, size_t topk, bool useSoftNms, float sigma, bool agnosticNMS){
    std::map<size_t, std::vector<DetectBox>> classifiedInfo;
    std::vector<DetectBox> bestBoxes;
    if(agnosticNMS){
      for(auto& i: info){
        classifiedInfo[i.imageId].push_back(i);
      }
    }else{
      for(auto& i: info){
        classifiedInfo[i.category].push_back(i);
      }
    }

    for (auto& ci: classifiedInfo) {
        auto& boxes = ci.second;
        std::sort(boxes.begin(), boxes.end(), [](DetectBox &a, DetectBox &b) {
          return b.confidence < a.confidence;
        });

        while(!boxes.empty()){
            auto bestBox = boxes[0];
            bestBoxes.push_back(bestBox);
            if(topk>0 && bestBoxes.size()>=topk){
                break;
            }
            if(!useSoftNms){
                boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&bestBox, iouThresh](const DetectBox& box){
                    return bestBox.iou(box) > iouThresh;
                }), boxes.end());
            } else {
                boxes.erase(boxes.begin());
                std::for_each(boxes.begin(), boxes.end(), [&bestBox, sigma](DetectBox& box){
                    auto iouScore = bestBox.iou(box);
                    auto weight = exp(-(1.0 * iouScore*iouScore / sigma));
                    box.confidence *= weight;
                });
            }
        }
    }
    return bestBoxes;
}

std::vector<DetectBox> singleNMS_agnostic(const std::vector<DetectBox> &info, float iouThresh, size_t topk, bool useSoftNms, float sigma){
    std::vector<DetectBox> classifiedInfo;
    std::vector<DetectBox> bestBoxes;
    for(auto& i: info){
      classifiedInfo.push_back(i);
    }
    auto& boxes = classifiedInfo;
    std::sort(boxes.begin(), boxes.end(), [](DetectBox &a, DetectBox &b) {
      return b.confidence < a.confidence;
    });
    while(!boxes.empty()){
      auto bestBox = boxes[0];
      bestBoxes.push_back(bestBox);
      int i = 0;
      while(i<boxes.size()){
        if(bestBox.iou(boxes[i]) > iouThresh){
          boxes.erase(boxes.begin()+i);
        }else{
          i++;
        }
      }
    }
    return bestBoxes;
}


using namespace jsonxx;
// return: map<image name, vector of detected bbox>
std::map<std::string, std::vector<DetectBox> > readCocoDatasetBBox(const std::string& cocoAnnotationFile)
{
    BMLOG(INFO, "Parsing annotation %s", cocoAnnotationFile.c_str());
    std::map<size_t, std::string> idToName;
    std::ifstream ifs(cocoAnnotationFile);
    Object coco;
    coco.parse(ifs);
    auto& images = coco.get<Array>("images");
    auto& annotations = coco.get<Array>("annotations");
    for(size_t i=0; i<images.size(); i++){
        auto& image = images.get<Object>(i);
        auto filename = image.get<std::string>("file_name");
        size_t id = image.get<Number>("id");
        idToName[id] = filename;
    }
    auto& categories = coco.get<Array>("categories");
    std::map<size_t, std::string> categoryMap;
    for(size_t i=0; i<categories.size(); i++){
        auto& category = categories.get<Object>(i);
        size_t id = category.get<Number>("id");
        auto name = category.get<std::string>("name");
        categoryMap[id]=name;
        BMLOG(INFO, "  category #%d: %s", id, name.c_str());
    }
    std::map<std::string, std::vector<DetectBox>> imageToBoxes;
    for(size_t i=0; i<annotations.size(); i++){
        auto& annotation = annotations.get<Object>(i);
        size_t imageId = annotation.get<Number>("image_id");
        if(!idToName.count(imageId)) continue;

        size_t categoryId = annotation.get<Number>("category_id");
        auto& bbox = annotation.get<Array>("bbox");
        DetectBox box;
        box.confidence = -1;
        box.category = categoryId;
        box.categoryName = categoryMap[categoryId];
        box.xmin = bbox.get<Number>(0);
        box.ymin = bbox.get<Number>(1);
        box.xmax = box.xmin + bbox.get<Number>(2);
        box.ymax = box.ymin + bbox.get<Number>(3);
        imageToBoxes[idToName[imageId]].push_back(box); 
        // imageToBoxes[imageId].push_back(box);
        box.imageId = imageId;
    }
    BMLOG(INFO, "Parsing annotation %s done", cocoAnnotationFile.c_str());
    return imageToBoxes;
}

void drawDetectBoxEx(bm_image &bmImage, const std::vector<DetectBox> &boxes, const std::vector<DetectBox> &trueBoxes, const std::string &saveName)
{
    //Draw a rectangle displaying the bounding box
    cv::Mat cvImage;
    auto status =cv::bmcv::toMAT(&bmImage, cvImage, true);
    BM_ASSERT_EQ(status, BM_SUCCESS);

    size_t borderWidth = 2;
    if(!trueBoxes.empty()){
        BMLOG(INFO, "draw true box for '%s'", saveName.c_str());
        for(size_t i=0; i<trueBoxes.size(); i++){
            auto& box = trueBoxes[i];
            cv::rectangle(cvImage, cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax), cv::Scalar(0, 255, 0), borderWidth);

            //Get the label for the class name and its confidence
            std::string label;
            label = std::to_string(box.category);
            if(box.categoryName != ""){
                label += "-" + box.categoryName;
            }

            //Display the label at the top of the bounding box
            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, 2, &baseLine);
            auto top = std::max((int)box.ymax, labelSize.height);
            cv::putText(cvImage, label, cv::Point(box.xmin, top), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 2);
            BMLOG(INFO, "  box #%d: [%d, %d, %d, %d], %s", i,
                  (size_t)box.xmin, (size_t)box.ymin, (size_t)box.xmax, (size_t)box.ymax, label.c_str());
        }
    }
    BMLOG(INFO, "draw predicted box for '%s'", saveName.c_str());
    for(size_t i=0; i<boxes.size(); i++){
        auto& box = boxes[i];
        cv::rectangle(cvImage, cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax), cv::Scalar(0, 0, 255), borderWidth);

        //Get the label for the class name and its confidence
        std::string label = std::string(":") + cv::format("%.2f", box.confidence);
        if(box.categoryName != ""){
            label = std::string("-") + box.categoryName +  label;
        }
        label = std::to_string(box.category) + label;

        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        auto top = std::max((int)box.ymin, labelSize.height);
        cv::putText(cvImage, label, cv::Point(box.xmin, top), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
        BMLOG(INFO, "  box #%d: [%d, %d, %d, %d], %s", i,
              (size_t)box.xmin, (size_t)box.ymin, (size_t)box.xmax, (size_t)box.ymax,
              label.c_str());
    }
    static size_t saveCount=0;
    std::string fullPath = saveName;
    if(fullPath=="") {
        fullPath = std::string("00000000") + std::to_string(saveCount)+".jpg";
        fullPath = fullPath.substr(fullPath.size()-4-8);
    }
    cv::imwrite(fullPath, cvImage);

}

void drawDetectBox(bm_image &bmImage, const std::vector<DetectBox> &boxes, const std::string &saveName)   // Draw the predicted bounding box
{
    return drawDetectBoxEx(bmImage, boxes, {}, saveName);
}

std::ostream& operator<<(std::ostream &os, const DetectBox &box){
    std::string categoryName = box.categoryName;
    strReplaceAll(categoryName, " ", "_");
    os<<categoryName<<" ";
    if(box.confidence>=0){
        os<<box.confidence<<" ";
    }
    os<<box.xmin<<" ";
    os<<box.ymin<<" ";
    os<<box.xmax<<" ";
    os<<box.ymax;
    return os;
}

void readCocoDatasetInfo(const std::string &cocoAnnotationFile, std::map<std::string, size_t>& nameToId, std::map<std::string, size_t> &nameToCategory)
{
    BMLOG(INFO, "Parsing annotation %s", cocoAnnotationFile.c_str());
    std::ifstream ifs(cocoAnnotationFile);
    Object coco;
    coco.parse(ifs);
    auto& images = coco.get<Array>("images");
    auto& annotations = coco.get<Array>("annotations");
    for(size_t i=0; i<images.size(); i++){
        auto& image = images.get<Object>(i);
        auto filename = image.get<std::string>("file_name");
        size_t id = image.get<Number>("id");
        nameToId[filename] = id;
    }
    auto& categories = coco.get<Array>("categories");
    for(size_t i=0; i<categories.size(); i++){
        auto& category = categories.get<Object>(i);
        size_t id = category.get<Number>("id");
        auto name = category.get<std::string>("name");
        nameToCategory[name]=id;
        BMLOG(INFO, "%d: %s", id, name.c_str());
    }
    BMLOG(INFO, "readCocoDatasetInfo  %s done", cocoAnnotationFile.c_str());
}

void saveCocoResults(const std::vector<DetectBox> &results, const std::string& filename)
{
    std::ofstream ofs(filename);
    ofs<<"["<<std::endl;
    for(size_t i=0; i<results.size()-1; i++){
        ofs<<results[i].json()<<","<<std::endl;
    }
    ofs<<results[results.size()-1].json()<<std::endl;
    ofs<<"]";
}


std::map<std::string, size_t> readCocoDatasetImageIdMap(const std::string &cocoAnnotationFile)
{
    BMLOG(INFO, "Parsing annotation %s", cocoAnnotationFile.c_str());
    std::ifstream ifs(cocoAnnotationFile);
    Object coco;
    coco.parse(ifs);
    std::map<std::string, size_t> nameToId;
    auto& images = coco.get<Array>("images");
    auto& annotations = coco.get<Array>("annotations");
    for(size_t i=0; i<images.size(); i++){
        auto& image = images.get<Object>(i);
        auto filename = image.get<std::string>("file_name");
        size_t id = image.get<Number>("id");
        nameToId[filename] = id;
    }
    BMLOG(INFO, "readCocoDatasetImageIdMap %s done", cocoAnnotationFile.c_str());
    return nameToId;
}

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float mAP::cal_iou(const bm::DetectBox& b1, const bm::DetectBox& b2) {
  float w = std::min(b1.xmax, b2.xmax) - std::max(b1.xmin, b2.xmin);
  float h = std::min(b1.ymax, b2.ymax) - std::max(b1.ymin, b2.ymin);
  if (w <= 0 || h <= 0) {
    return 0.f;
  }

  float union_area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin) +
                             (b2.xmax - b2.xmin) * (b2.ymax - b2.ymin);
  float iou = (w * h) / (union_area - (w * h));
  std::cout<<"iou====================="<<iou<<std::endl;
  return iou;
}

// 输入是每一张图的框
void mAP::match(std::vector<bm::DetectBox>& preds, std::vector<bm::DetectBox>& refs){
  std::cout<<"predict num:"<<preds.size()<<"  refs num:"<<refs.size()<<std::endl;
  const float iou_thesh = 0.6f;
  for (size_t i = 0; i < preds.size(); i++) {
    for (size_t j = 0; j < refs.size(); j++) {
      if (preds[i].category != refs[j].category-1) {
        continue;
      }
      // 同一检测框两次命中，confidence低的同样作为fp
      if (cal_iou(preds[i], refs[j]) >= iou_thesh && !refs[j].matched) {
        preds[i].matched = true;
        refs[j].matched = true;
        std::cout<<"match!"<<std::endl;
        break;
      }
    }
  }
  for (size_t i = 0; i < refs.size(); i++) {
    if (!refs[i].matched) {
      fn_[refs[i].category] += 1;
    }
  }
  return;
}

// 输入是所有测试图片中某一类的框
float mAP::calc_ap(const std::vector<st_calc_ap_info>& preds){
  int tp = 0;
  int fp = 0;
  float ap = 0.f;
  float max_presions = 0.f;
  float recall = 0.f;
  float presion = 0.f;
  float recall_point = 0.f;
  float section_box_num = 0;
  for (int i = 0; i < preds.size(); i++) {
    if (preds[i].matched) {
      tp += 1;
    } else {
      fp += 1;
    }
    if (0 != tp + fn_[preds[i].class_id - 1]) {
      recall = (float)tp / ((float)tp + (float)fn_[preds[i].class_id - 1]);
    }
    if (0 != tp + fp) {
      presion = (float)tp / ((float)tp + (float)fp);
    }
    if (recall >= recall_point) {
      recall_point += 0.1;
      if (section_box_num != 0) {
        ap += max_presions / 11;
      }
      max_presions = presion;
      section_box_num = 1.f;
    } else {
      section_box_num += 1;
      if (presion > max_presions) {
        max_presions = presion;
      }
    }
  }
  return ap;
}

static bool sort_score(st_calc_ap_info r1, st_calc_ap_info r2) {
  return (r1.score > r2.score);
}

/*
  sorted_output: dict of every single image
*/
/*
float mAP::calc(std::map<int, std::vector<struct bm::DetectBox>>& sorted_outputs, const std::string &refFile) {
  std::cout<<"sorted_outputs size = "<<sorted_outputs.size()<<std::endl;  
  std::vector<st_calc_ap_info> perclass_preds[class_num_];
  for (int i = 0; i < class_num_; i++) {
    fn_.push_back(0);
  }

  std::map<size_t, std::vector<bm::DetectBox>> refs;
  refs = bm::readCocoDatasetBBox(refFile);

  // preds: std::vector<bm::DetectBox> boxes in each image
  for (auto preds : sorted_outputs) {
    int index = preds.first;
    match(preds.second, refs[index]);
    // 所有的框按类别分组
    for (size_t i = 0; i < preds.second.size(); i++) {
      // std::cout<<"======= i ======="<<i<<std::endl;
      st_calc_ap_info calc_info;
      calc_info.matched = preds.second[i].matched;
      calc_info.score = preds.second[i].confidence;
      calc_info.class_id = preds.second[i].imageId;
      perclass_preds[preds.second[i].category].push_back(calc_info);
    }
  }

  // 每一类别的所有框按score降序排列
  float total_ap = 0.f;
  for (int i = 0; i < class_num_; i++) {
    std::sort(perclass_preds[i].begin(), perclass_preds[i].end(), sort_score);
    auto ap = calc_ap(perclass_preds[i]);
    std::cout<<"class "<<i<<" ap ======="<<ap<<std::endl;
    total_ap += ap;
  }
  float mAP_val = total_ap / class_num_;
  return mAP_val;
}
*/
