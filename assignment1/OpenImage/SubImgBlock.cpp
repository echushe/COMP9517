#include "SubImgBlock.h"



SubImgBlock::SubImgBlock(int top, int left, int height, int width, int o_height, int o_width, SCNPixelPtr *matrix)
    : m_top(top),
    m_left(left),
    m_height(height),
    m_width(width),
    m_o_height(o_height),
    m_o_width(o_width),
    m_matrix(matrix)
{
    if (this->m_top < 0)
    {
        this->m_top += this->m_o_height;
    }

    if (this->m_left < 0)
    {
        this->m_left += this->m_o_width;
    }

    if (this->m_top >= this->m_o_height)
    {
        this->m_top -= this->m_o_height;
    }

    if (this->m_left >= this->m_o_width)
    {
        this->m_left -= this->m_o_width;
    }
}


SubImgBlock::~SubImgBlock()
{
}

SCNPixel & SubImgBlock::pixelAt(int row, int col) const
{
    if (col >= this->m_width || col < 0 || row >= this->m_height || row < 0)
    {
        throw "Index out of range";
    }

    int real_row = this->m_top + row;
    int real_col = this->m_left + col;

    if (real_row < 0)
    {
        real_row += this->m_o_height;
    }

    if (real_col < 0)
    {
        real_col += this->m_o_width;
    }

    if (real_row >= this->m_o_height)
    {
        real_row -= this->m_o_height;
    }

    if (real_col >= this->m_o_width)
    {
        real_col -= this->m_o_width;
    }

    return *(this->m_matrix[real_row * this->m_o_width + real_col]);
}

double SubImgBlock::imageDistance(const SubImgBlock & other)
{
    if (other.m_height != this->m_height || other.m_width != this->m_width)
    {
        throw "Sizes of these images are different!";
    }

    double sum = 0.0;
    double my_mean = 0.0;
    double your_mean = 0.0;

    
    for (int i = -1, j = 0;; j++)
    {
        j %= this->m_width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->m_height)
        {
            break;
        }
        
        sum += this->pixelAt(i, j).getGray();
    }

    my_mean = sum / (this->m_width * this->m_height);

    sum = 0.0;
    for (int i = -1, j = 0;; j++)
    {
        j %= this->m_width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->m_height)
        {
            break;
        }
        
        sum += other.pixelAt(i, j).getGray();
    }

    your_mean = sum / (this->m_width * this->m_height);

    double mean_offset = your_mean - my_mean;

    sum = 0.0;
    for (int i = 0; i < this->m_height; i++)
    {
        for (int j = 0; j < this->m_width; j++)
        {
            uchar left = this->pixelAt(i, j).getGray();
            uchar right = other.pixelAt(i, j).getGray();
            double offset = left + mean_offset - right;

            sum += offset * offset;
        }
    }

    return sum / (this->m_width * this->m_height);
}

double SubImgBlock::getGrayVariance() const
{
    double sum = 0.0;
    double mean = 0.0;

    for (int i = -1, j = 0;; j++)
    {
        j %= this->m_width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->m_height)
        {
            break;
        }
        
        sum += this->pixelAt(i, j).getGray();
    }

    mean = sum / (this->m_width * this->m_height);

    sum = 0.0;
    for (int i = -1, j = 0;; j++)
    {
        j %= this->m_width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->m_height)
        {
            break;
        }
        
        uchar gray_color = this->pixelAt(i, j).getGray();
        double offset = mean - gray_color;

        sum += offset * offset;
    }

    return sum / (this->m_width * this->m_height);
}
