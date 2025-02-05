/***
 * Function: 360 surrond view combine c++ cuda acc demo
 * Author: FunnyWii
 * Date: Apr.12.2024
 * Copyright: TSARI all right reserved
 */

#include "common.h"
#include <chrono>

// #define DEBUG
#define AWB_LUN_BANLANCE_ENALE 0

int main(int argc, char **argv)
{
    std::cout << argv[0] << " app start running..." << std::endl;
    cv::Mat car_img;
    cv::Mat origin_dir_img[4];
    cv::Mat undist_dir_img[4];
    cv::Mat merge_weights_img[4];
    cv::Mat out_put_img;
    float *w_ptr[4];
    CameraPrms prms[4];

    // 1.read image and read weights
    car_img = cv::imread("../images/car.png");
    cv::resize(car_img, car_img, cv::Size(xr - xl, yb - yt));
    out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat weights = cv::imread("../yaml/weights.png", -1);

    if (weights.channels() != 4)
    {
        std::cerr << "imread weights failed " << weights.channels() << "\r\n";
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        merge_weights_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
        w_ptr[i] = (float *)merge_weights_img[i].data;
    }
    // read weights of corner
    int pixel_index = 0;
    for (int h = 0; h < weights.rows; ++h)
    {
        uchar *uc_pixel = weights.data + h * weights.step;
        for (int w = 0; w < weights.cols; ++w)
        {
            w_ptr[0][pixel_index] = uc_pixel[0] / 255.0f;
            w_ptr[1][pixel_index] = uc_pixel[1] / 255.0f;
            w_ptr[2][pixel_index] = uc_pixel[2] / 255.0f;
            w_ptr[3][pixel_index] = uc_pixel[3] / 255.0f;
            uc_pixel += 4;
            ++pixel_index;
        }
    }

#ifdef DEBUG
    for (int i = 0; i < 4; ++i)
    {
        // 0 左下 1 右上 2 左上 3 左下
        display_mat(merge_weights_img[i], "w");
    }
#endif

    // 1. read calibration prms
    for (int i = 0; i < 4; ++i)
    {
        auto &prm = prms[i];
        prm.name = camera_names[i];
        auto ok = read_prms("../yaml/" + prm.name + ".yaml", prm);
        if (!ok)
        {
            return -1;
        }
    }

    // 2.lum equalization and awb for four channel image
    std::vector<cv::Mat *> srcs;
    for (int i = 0; i < 4; ++i)
    {
        auto &prm = prms[i];
        origin_dir_img[i] = cv::imread("../images/" + prm.name + ".png");
        srcs.push_back(&origin_dir_img[i]);
    }

#if AWB_LUN_BANLANCE_ENALE
    awb_and_lum_banlance(srcs);
#endif
    //  "front", "left", "back", "right"
    cv::VideoCapture CapF, CapB, CapL, CapR;
    CapF.open(2);
    CapB.open(3);
    CapL.open(0);
    CapR.open(1);
    cv::Mat frameF, frameB, frameL, frameR;
    std::vector<std::pair<cv::Mat, cv::Mat>> map;
    // std::vector<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> mapG;
    std::vector<cv::cuda::GpuMat> mapG1, mapG2;
    std::vector<cv::cuda::GpuMat> prmG;
    std::vector<cv::cuda::GpuMat> frameG;
    std::vector<cv::Size> sizeG;
    for (int i = 0; i < 4; ++i){
        auto &prm = prms[i];
        cv::cuda::GpuMat temp(prms[i].project_matrix);
        prmG.push_back(temp);
        cv::Mat &src = origin_dir_img[i];
        cv::cuda::GpuMat tempImg(origin_dir_img[i]);
        frameG.push_back(tempImg);
        sizeG.push_back(project_shapes[prms[i].name]);

        map.push_back(undist_by_remap(src, src, prm));
        cv::Mat map1 = map[i].first;
        cv::Mat map2 = map[i].second;
        map1.convertTo(map1,CV_32FC1);
        map2.convertTo(map2,CV_32FC1);

        cv::cuda::GpuMat temp1(map1);
        cv::cuda::GpuMat temp2(map2);
        // temp1.convertTo(temp1,CV_32FC1);
        // temp2.convertTo(temp2,CV_32FC1);
        if (temp1.type() !=5 || temp2.type() !=5){
            perror("Map matrix type must be CV_32FC1!!!");
            return -1;
        }
        mapG1.push_back(temp1);
        mapG2.push_back(temp2);

    }

    while (CapF.grab() && CapB.grab() && CapL.grab() && CapR.grab())
    {
        auto start = std::chrono::high_resolution_clock::now();

        CapF.retrieve(frameF);
        CapB.retrieve(frameB);
        CapL.retrieve(frameL);
        CapR.retrieve(frameR);
        origin_dir_img[0] = frameF;
        origin_dir_img[1] = frameL;
        origin_dir_img[2] = frameB;
        origin_dir_img[3] = frameR;

        // }
        // 3. undistort image
        for (int i = 0; i < 4; ++i){

            auto &prm = prms[i];
            cv::Mat &src = origin_dir_img[i];

            cv::cuda::GpuMat srcG(src);
            cv::cuda::remap(srcG, srcG, mapG1[i], mapG2[i], cv::INTER_NEAREST, cv::BORDER_CONSTANT);
            // cv::remap(src, src, map[i].first, map[i].second, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
            // srcG.download(src);

            cv::cuda::warpPerspective(srcG, srcG, prmG[i], sizeG[i]);
            // cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);
         
            if (camera_flip_mir[i] == "r+")
            {
                cv::cuda::rotate(srcG, srcG, srcG.size(), cv::ROTATE_90_CLOCKWISE);
                // cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
            }
            else if (camera_flip_mir[i] == "r-")
            {
                cv::cuda::rotate(srcG, srcG, srcG.size(),cv::ROTATE_90_COUNTERCLOCKWISE);
                // cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
            else if (camera_flip_mir[i] == "m")
            {
                cv::cuda::rotate(srcG, srcG, srcG.size(),cv::ROTATE_180);
                // cv::rotate(src, src, cv::ROTATE_180);
            }
            // display_mat(src, "project");
            // cv::imwrite(prms.name + "_undist.png", src);
            srcG.download(src);

            undist_dir_img[i] = src.clone();
        }

        // 4.start combine
        std::cout << argv[0] << " app start combine" << std::endl;
        car_img.copyTo(out_put_img(cv::Rect(xl, yt, car_img.cols, car_img.rows)));
        // 4.1 out_put_img center copy
        for (int i = 0; i < 4; ++i)
        {
            cv::Rect roi;
            bool is_cali_roi = false;
            if (std::string(camera_names[i]) == "front")
            {
                roi = cv::Rect(xl, 0, xr - xl, yt);
                // std::cout << "\nfront" << roi;
                undist_dir_img[i](roi).copyTo(out_put_img(roi));
            }
            else if (std::string(camera_names[i]) == "left")
            {
                roi = cv::Rect(0, yt, xl, yb - yt);
                // std::cout << "\nleft" << roi << out_put_img.size();
                undist_dir_img[i](roi).copyTo(out_put_img(roi));
            }
            else if (std::string(camera_names[i]) == "right")
            {
                roi = cv::Rect(0, yt, xl, yb - yt);
                // std::cout << "\nright" << roi << out_put_img.size();
                undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
            }
            else if (std::string(camera_names[i]) == "back")
            {
                roi = cv::Rect(xl, 0, xr - xl, yt);
                // std::cout << "\nright" << roi << out_put_img.size();
                undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
            }
        }
        // 4.2the four corner merge
        // w: 0 左下 1 右上 2 左上 3 左下
        // image: "front", "left", "back", "right"
        cv::Rect roi;
        // 左上
        roi = cv::Rect(0, 0, xl, yt);
        merge_image(undist_dir_img[0](roi), undist_dir_img[1](roi), merge_weights_img[2], out_put_img(roi));

        // 右上
        roi = cv::Rect(xr, 0, xl, yt);
        merge_image(undist_dir_img[0](roi), undist_dir_img[3](cv::Rect(0, 0, xl, yt)), merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
        // 左下
        roi = cv::Rect(0, yb, xl, yt);
        merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](roi), merge_weights_img[0], out_put_img(roi));
        // 右下
        roi = cv::Rect(xr, 0, xl, yt);
        merge_image(undist_dir_img[2](roi), undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3], out_put_img(cv::Rect(xr, yb, xl, yt)));

        // cv::imwrite("./images/ADAS_EYES_360_VIEW.png", out_put_img);

#ifdef DEBUG
        cv::resize(out_put_img, out_put_img, cv::Size(out_put_img.size() / 2)),
            display_mat(out_put_img, "out_put_img");
#endif

        std::cout << argv[0] << " app finished" << std::endl;
        cv::Rect finalROI = cv::Rect(200, 200, 460, 460);
        cv::Size finalSize(920,920);
        cv::Mat finalOutput;
        cv::resize(out_put_img(finalROI),finalOutput,finalSize,cv::INTER_LINEAR);
        
        cv::imshow("AVM SYSTEM", finalOutput);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Elapsed time: " << duration << "millisecond" << std::endl;
        cv::waitKey(40);
    }
    return 0;
}