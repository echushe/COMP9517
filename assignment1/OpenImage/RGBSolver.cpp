#include "RGBSolver.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "SCNImage.h"
#include "SCNPixel.h"
#include <iostream>

using namespace cv;

RGBSolver::RGBSolver(std::string file_name, int window_size, int image_pyramid_start_size)
{
    Mat image;
    image = imread(file_name, CV_LOAD_IMAGE_COLOR);   // Read the file

    if (!image.data)                              // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return;
    }

    // SCNImage my_img(image);
    SCNImage my_combined[6];
    // my_img.reverseColor();
    SCNImage::getCombinedImages(image, window_size, image_pyramid_start_size, my_combined);
    SCNImage my_modified[6];

    for (int i = 0; i < 6; i++)
    {
        Mat l_image;
        my_combined[i].toCVImage(l_image);
        // imshow("Display window", image);                   // Show our image inside it.
        std::string output_file_name = file_name + std::to_string(i) + "_output.jpg";
        imwrite(output_file_name, l_image);

        int height = my_combined[i].getHeight();
        int width = my_combined[i].getWidth();
        my_combined[i].subImage(height / 20, width / 20, height * 18 / 20, width * 18 / 20, my_modified[i]);

        my_modified[i].histogramEqualization();
        my_modified[i].toCVImage(l_image);
        // imshow("Display window", image);                   // Show our image inside it.
        output_file_name = file_name + std::to_string(i) + "_output_hg.jpg";
        imwrite(output_file_name, l_image);

        my_combined[i].destroyMe();
        my_modified[i].destroyMe();
    }
}


RGBSolver::~RGBSolver()
{
}
