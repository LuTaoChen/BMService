#ifndef SEGMENTATIONUTILS_H
#define SEGMENTATIONUTILS_H

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "bmcv_api.h"

namespace bm
{

    template <typename T, typename Pred = std::function<T(const T &)>>
    int argmax(
        const T *data,
        size_t len,
        size_t stride = 1,
        Pred pred = [](const T &v)
        { return v; })
    {
        int maxIndex = 0;
        for (size_t i = 1; i < len; i++)
        {
            int idx = i * stride;
            if (pred(data[maxIndex * stride]) < pred(data[idx]))
            {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * @brief
     *
     * @param result Must specified with width and height first, must be rgb
     *                 result.rows * result.cols must equal to length of cls_per_pix
     * @param cls_per_pix category per pixel, length = output_h * output_w
     *                       cls_per_pix must equal to length of result.rows * result.cols
     * @param palette bgr color
     */
    void SegmentationVisualization(cv::Mat result, uchar *cls_per_pix, cv::Vec3b palette[]);

    /* compare pixel */
    struct vec3b_cmp
    {
        bool operator()(const cv::Vec3b &v1, const cv::Vec3b &v2) const
        {
            for (int i = 0; i < v1.channels; i++)
            {
                if (v1[i] != v2[i])
                {
                    return v1[i] < v2[i];
                }
            }
            return false;
        }
    };

    class ConfusionMatrix
    {
    public:
        ConfusionMatrix(int cls_num, cv::Vec3b *palette);
        ~ConfusionMatrix();

        /**
         * @brief Update confusion matrix summation over images seen so far 
         *          and return confusion matrix of single image
         * 
         * @param gt 
         * @param pred array of category, length = w * h
         * @return std::vector<std::vector<int>> confusion matrix of this image
         */
        std::vector<std::vector<int>> update(cv::Mat &gt, uchar *pred);

        std::vector<std::vector<int>> update(cv::Mat &gt, cv::Mat &pred);
        int ** get_confusion_mat();
        int * get_TP();
        int get_white();

        template <typename T>
        std::vector<float> cal_mIoU(T &conf_mat)
        {
            std::vector<float> IoU;
            float mIoU = 0;
            cal_TPTNFPFN(conf_mat);
            int cls_in_img = 0;
            for (int i = 0; i < class_num; i++)
            {
                int u = TP[i] + FN[i] + FP[i];
                if (u != 0)
                {
                    cls_in_img++;
                    IoU.push_back((float)TP[i] / (float)u);
                    mIoU += IoU.at(IoU.size()-1);
                } else {
                    IoU.push_back(0);
                }
            }
            IoU.push_back(mIoU / cls_in_img);
            return IoU;
        }        

        /**
         * @brief 
         * 
         * @tparam T vector of vectors of int or 2-d array
         * @param conf_mat row: each prediction class, col: each ground truth class
         */
        template <typename T>
        void cal_TPTNFPFN(T &conf_mat)
        {
            int all = 0;

            // init
            for (int i = 0; i < class_num; i++)
            {
                TP[i] = 0;
                FP[i] = 0;
                FN[i] = 0;
                TN[i] = 0;
            }

            for (int r = 0; r < class_num; r++)
            {
                for (int c = 0; c < class_num; c++)
                {
                    if (r == c)
                    {
                        TP[r] += conf_mat[r][c]; // TP[c] = conf_mat[r][c]
                    }
                    else
                    {
                        FP[r] += conf_mat[r][c];
                        FN[c] += conf_mat[r][c];
                    }
                    all += conf_mat[r][c];
                }
            }

            for (int i = 0; i < class_num; i++)
            {
                TN[i] = all - (TP[i] + FN[i] + FP[i]);
            }
        }
    private:
        int class_num;
        int **conf_mat_all; /* row: each predict, col: each ground truth */
        int correct;
        std::map<cv::Vec3b, int, vec3b_cmp> pix2cls;
        cv::Vec3b *palette;
        int *TP, *TN, *FP, *FN;
        int white;
    };

}
#endif // SEGMENTATIONUTILS_H
