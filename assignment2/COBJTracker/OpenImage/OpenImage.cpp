#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "XContour.h"
#include <cstring>

//using namespace cv;
//using namespace cv::xfeatures2d;
//using namespace std;

/*
    // Read first frame. 
    Mat frame;
    video.read(frame);

    // Define an initial bounding box
    Rect2d bbox(287, 23, 86, 320);

    // Uncomment the line below if you 
    // want to choose the bounding box
    // bbox = selectROI(frame, false);
*/

void draw_rects_on_image(cv::Mat &img, std::vector<cv::Rect> &rects)
{
    for (cv::Rect rect : rects)
    {
        cv::rectangle(img, rect, cv::Scalar(0.0, 0.0, 255.0), 2);
    }
}

void sub_images_via_rect(const cv::Mat &img, const std::vector<cv::Rect> &rects, std::vector<cv::Mat> &sub_imgs)
{
    sub_imgs.clear();
    for (cv::Rect rect : rects)
    {
        cv::Mat sub_img{ img, rect };
        sub_imgs.push_back(sub_img);
    }
}


void get_contours_via_two_frames(const cv::Mat & frame_1, const cv::Mat & frame_2, cv::Mat & threshold, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Rect> &rects)
{
    cv::Mat l_frame_1 = frame_1.clone();
    cv::Mat l_frame_2 = frame_2.clone();

    cv::Mat difference;
    
    // Convert the frames to gray color
    cv::cvtColor(l_frame_1, l_frame_1, CV_BGR2GRAY);
    cv::cvtColor(l_frame_2, l_frame_2, CV_BGR2GRAY);

    // Blur
    cv::GaussianBlur(l_frame_1, l_frame_1, cv::Size(5, 5), 0);
    cv::GaussianBlur(l_frame_2, l_frame_2, cv::Size(5, 5), 0);

    //
    cv::absdiff(l_frame_1, l_frame_2, difference);
    cv::threshold(difference, threshold, 5, 255.0, CV_THRESH_BINARY);

    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    
    for (unsigned int i = 0; i < 2; i++)
    {
        cv::dilate(threshold, threshold, structuringElement5x5);
        cv::dilate(threshold, threshold, structuringElement5x5);
        cv::erode(threshold, threshold, structuringElement5x5);
    }

    std::vector<std::vector<cv::Point>> possible_contours;
    cv::findContours(threshold, possible_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto contour : possible_contours)
    {
        std::vector<cv::Point> convexHull;
        cv::convexHull(contour, convexHull);
        XContour xc(convexHull);

        if (xc.currentBoundingRect.area() > 400 &&
            xc.dblCurrentAspectRatio > 0.2 &&
            xc.dblCurrentAspectRatio < 4.0 &&
            xc.currentBoundingRect.width > 30 &&
            xc.currentBoundingRect.height > 30 &&
            xc.dblCurrentDiagonalSize > 200.0 &&
            (cv::contourArea(xc.currentContour) / (double)xc.currentBoundingRect.area()) > 0.50)
        {
            contours.push_back(convexHull);
        }
    }

    for (std::vector<cv::Point> contour : contours)
    {
        cv::Rect rect{ cv::boundingRect(contour) };
        rects.push_back(rect);
    }
}

// Neither of the two frames are detected or computed
void match_2_frames(
    cv::Mat & img_1, 
    cv::Mat & img_2,
    cv::Ptr<cv::xfeatures2d::SURF> detector, int index)
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(img_1, cv::Mat{}, keypoints1, descriptors1);
    detector->detectAndCompute(img_2, cv::Mat{}, keypoints2, descriptors2);

    std::cout << keypoints1.size() << std::endl;

    std::cout << keypoints2.size() << std::endl;

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches;

    auto flann = cv::FlannBasedMatcher{};
    flann.match(descriptors1, descriptors2, matches);


    for (cv::DMatch match : matches)
    {
        if (match.distance < 0.2)
        {
            good_matches.push_back(match);
        }
    }

    if (good_matches.size() < 1)
    {
        return;
    }
    // drawing the results
    cv::Mat img_matches;

    // drawMatches(img_1, keypoints1, img_2, keypoints2, good_matches, img_matches);

    cv::drawMatches(img_1, keypoints1, img_2, keypoints2,
        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches: " + index, img_matches);
}

void match_2_frames_GPU(
    cv::Mat & img_1,
    cv::Mat & img_2,
    cv::Ptr<cv::xfeatures2d::SURF> detector, int index)
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::cuda::GpuMat img_1_gpu, img_2_gpu;
    img_1_gpu.upload(img_1);
    img_2_gpu.upload(img_2);

    cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
    cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;

    cv::cuda::SURF_CUDA surf(400);
    surf(img_1_gpu, cv::cuda::GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img_2_gpu, cv::cuda::GpuMat(), keypoints2GPU, descriptors2GPU);

    std::cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << std::endl;
    std::cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << std::endl;

    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    keypoints1GPU.download(descriptors1);
    keypoints2GPU.download(descriptors2);
    
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches;

    auto flann = cv::FlannBasedMatcher{};
    flann.match(descriptors1, descriptors2, matches);


    for (cv::DMatch match : matches)
    {
        if (match.distance < 0.2)
        {
            good_matches.push_back(match);
        }
    }

    if (good_matches.size() < 1)
    {
        return;
    }
    // drawing the results
    cv::Mat img_matches;

    // drawMatches(img_1, keypoints1, img_2, keypoints2, good_matches, img_matches);

    cv::drawMatches(img_1, keypoints1, img_2, keypoints2,
        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches: " + index, img_matches);
}


int track_video(std::string video_name, int obj_id)
{
    // declares all required variables
    cv::Mat frame_1;
    cv::Mat frame_2;
    cv::Mat frame_3;
    cv::Mat frame_4;
    cv::Mat frame_1_c;
    cv::Mat l_threshold;
    cv::Mat r_threshold;

    std::vector<cv::Mat> l_sub_images;
    std::vector<cv::Mat> r_sub_images;

    // set input video
    cv::VideoCapture cap(video_name);

    if (cap.get(CV_CAP_PROP_FRAME_COUNT) < 4)
    {
        std::cout << "error: video file must have at least four frames";
        return 0;
    }

    for (int i = 0; ; i++)
    {
        // get bounding box
        cap >> frame_1;
        cap >> frame_2;

        cv::Mat _frame_1_c(frame_1.rows, frame_1.cols, CV_8U);
        frame_1_c = _frame_1_c;

        // stop the program if no more images
        if (frame_1.rows == 0 || frame_1.cols == 0 || frame_2.rows == 0 || frame_2.cols == 0)
        {
            return 0;
        }

        std::vector<std::vector<cv::Point>> left_contours;
        std::vector<cv::Rect> left_rects;
        get_contours_via_two_frames(frame_1, frame_2, l_threshold, left_contours, left_rects);

        if (left_rects.size() > 2)
        {   
            cv::drawContours(frame_1_c, left_contours, -1, cv::Scalar(255.0, 255.0, 255.0), -1);
            sub_images_via_rect(frame_1, left_rects, l_sub_images);

            draw_rects_on_image(frame_1, left_rects);
            imshow("L Objects detected:", frame_1);
            std::cout << frame_1.size() << std::endl;
            std::cout << frame_1_c.size() << std::endl;
            break;
        }
    }

    // surf detector detecting keypoints
    int minHessian = 600;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

    printf("Start the tracking process, press ESC to quit.\n");
    for (int i = 0; l_sub_images.size() > obj_id; i++)
    {
        // get frame from the video
        cap >> frame_3;
        cap >> frame_4;

        // stop the program if no more images
        if (frame_3.rows == 0 || frame_3.cols == 0 || frame_4.rows == 0 || frame_4.cols == 0)
        {
            break;
        }

        std::vector<std::vector<cv::Point>> right_contours;
        std::vector<cv::Rect> right_rects;
        get_contours_via_two_frames(frame_3, frame_4, r_threshold, right_contours, right_rects);
        sub_images_via_rect(frame_3, right_rects, r_sub_images);
        // draw_rects_on_image(frame_3, right_rects);
        // cv::drawContours(frame_3, right_contours, -1, cv::Scalar(255.0, 255.0, 255.0), -1);
        // cv::imshow("R Objects detected:", frame_3);

        //for (int i = 0; i < l_sub_images.size(); i++)
        {
            //for (int j = 0; j < r_sub_images.size(); j++)
            {
                match_2_frames(l_sub_images[obj_id], frame_3, detector, 0);
            }
        }

        // imshow("R Objects detected:", frame_3);

        if (cv::waitKey(1) == 27)break;
    }
    return 0;
}


int main(int argc, char** argv)
{
    if (argc == 3)
    {
        return track_video(argv[1], std::atoi(argv[2]));
    }
}

/** surf main */


