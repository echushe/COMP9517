#pragma once
#include "./SCNPixel.h"
typedef SCNPixel *SCNPixelPtr;

class SubImgBlock
{
private:
    int m_height;
    int m_width;
    int m_top;
    int m_left;
    int m_o_height;
    int m_o_width;

    SCNPixelPtr *m_matrix;

public:
    SubImgBlock(int top, int left, int height, int width, int o_height, int o_width, SCNPixelPtr *matrix);
    ~SubImgBlock();

public:
    SCNPixel & pixelAt(int row, int col) const;
    double imageDistance(const SubImgBlock &other);
    double getGrayVariance() const;
};

