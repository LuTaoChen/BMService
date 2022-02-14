#include <fstream>
#include "jsonxx.h"
#include "opencv2/opencv.hpp"
#include "BMCommonUtils.h"
#include "SegmentationUtils.h"
#include "BMLog.h"

namespace bm
{

    void SegmentationVisualization(cv::Mat result, uchar *cls_per_pix, cv::Vec3b palette[])
    {
        for (int i = 0; i < result.rows * result.cols; i++)
        {
            auto cls_num = cls_per_pix[i];
            int row = i / result.cols;
            int col = i % result.cols;
            result.at<cv::Vec3b>(row, col)[0] = palette[cls_num][0];
            result.at<cv::Vec3b>(row, col)[1] = palette[cls_num][1];
            result.at<cv::Vec3b>(row, col)[2] = palette[cls_num][2];
        }
    }

    ConfusionMatrix::ConfusionMatrix(int cls_num, cv::Vec3b *palette)
    {
        this->palette = palette;
        class_num = cls_num;
        conf_mat_all = new int *[cls_num]();
        for (int i = 0; i < cls_num; i++)
        {
            conf_mat_all[i] = new int[cls_num]();
            pix2cls[palette[i]] = i;
        }
        // since white line in label image doesn't belong to any category
        // pix2cls[cv::Vec3b(255, 255, 255)] = 0;

        TP = new int[cls_num];
        TN = new int[cls_num];
        FP = new int[cls_num];
        FN = new int[cls_num];
    }

    ConfusionMatrix::~ConfusionMatrix()
    {
        delete[] conf_mat_all;
        delete TP;
        delete TN; 
        delete FP;
        delete FN;
    }


    std::vector<std::vector<int>> ConfusionMatrix::update(cv::Mat &gt, uchar *pred)
    {
        int area = gt.rows * gt.cols;
        white = 0;
        std::vector<std::vector<int>> conf_mat(class_num, std::vector<int>(class_num, 0));
        for (int i = 0; i < gt.rows * gt.cols; i++)
        {
            int row = i / gt.cols;
            int col = i % gt.cols;
            int pred_cls = (int)pred[i];
            auto key = gt.at<cv::Vec3b>(row, col);
            // ignore white
            if (pix2cls.count(key) == 1) 
            {
                int gt_cls = pix2cls[key];
                conf_mat[pred_cls][gt_cls]++;
            }
            else 
            {
                white++;
            }
        }

        // update all
        for (int r = 0; r < class_num; r++)
        {
            for (int c = 0; c < class_num; c++)
            {
                conf_mat_all[r][c] += conf_mat[r][c];
            }
        }
        return conf_mat;
    }


    std::vector<std::vector<int>> ConfusionMatrix::update(cv::Mat &gt, cv::Mat &pred)
    {
        assert(gt.rows == pred.rows);
        assert(gt.cols == pred.cols);

        int area = gt.rows * gt.cols;

        white = 0;
        std::vector<std::vector<int>> conf_mat(class_num, std::vector<int>(class_num, 0));
        for (int i = 0; i < gt.rows * gt.cols; i++)
        {
            int row = i / gt.cols;
            int col = i % gt.cols;
            int pred_cls = pix2cls[pred.at<cv::Vec3b>(row, col)];
            auto key = gt.at<cv::Vec3b>(row, col);
            // ignore white
            if (pix2cls.count(key) == 1) 
            {
                int gt_cls = pix2cls[key];
                conf_mat[pred_cls][gt_cls]++;
            }
            else 
            {
                white++;
            }
        }

        // update all
        for (int r = 0; r < class_num; r++)
        {
            for (int c = 0; c < class_num; c++)
            {
                conf_mat_all[r][c] += conf_mat[r][c];
            }
        }
        return conf_mat;
    }


    int **ConfusionMatrix::get_confusion_mat()
    {
        return conf_mat_all;
    }

    int * ConfusionMatrix::get_TP() 
    {
        return TP;
    }
    int ConfusionMatrix::get_white() 
    {
        return white;
    }


} // end of bm namespace