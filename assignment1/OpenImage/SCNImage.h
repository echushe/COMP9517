#pragma once
#include <vector>
#include "./SCNPixel.h"
#include "opencv2/core/core.hpp"

typedef SCNPixel *SCNPixelPtr;
using namespace cv;

class MatchWindow;

class SCNImage
{
private:
    int width;
    int height;

    SCNPixelPtr *m_matrix;

public:
    SCNImage();
    SCNImage(const Mat &cv_img);
    SCNImage(int h, int w);
    SCNImage(const SCNImage &other);
    ~SCNImage();

public:
    SCNImage & operator = (const SCNImage &other);

    int getWidth() const
    {
        return this->width;
    }

    int getHeight() const
    {
        return this->height;
    }

    void reverseColor();
    void histogramEqualization();

    SCNPixel & pixelAt(int row, int col) const;
    void setPixelAt(int row, int col, uchar gray);
    void setPixelAt(int row, int col, uchar r, uchar g, uchar b);
    void subImage(int start_row, int start_col, int h, int w, SCNImage &dest) const;
    void toCVImage(Mat &cv_img);
    double getGrayVariance() const;
    void destroyMe();

public:
    static void getCombinedImages(const Mat &cv_img, int window_size, int pyramid_start_size,  SCNImage *combined);
    static void match(const SCNImage &img, int size, MatchWindow& win0, MatchWindow& win1, MatchWindow& win2);
    static void match(const SCNImage & img, int size, int row, int col, int dest_row_start, int dest_col_start, MatchWindow & dest);

private:
    static void combineImage(int rgb_ids[3], const Mat &cv_img, int start_row, int start_col, int dest1_row_start, int dest1_col_start, int dest2_row_start, int dest2_col_start, SCNImage &combined);
    static void randomInitialMatchPoint(const SCNImage &img, int size, int &row, int &col);
    double imageDistance(const SCNImage &other);
    static bool overlap(int xs1[], int ys1[], int xs2[], int ys2[]);
};

