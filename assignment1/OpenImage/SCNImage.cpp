#include "SCNImage.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "./MatchWindow.h"
#include "./SubImgBlock.h"
#include <queue>
#include <stack>
#include <vector>
#include <iostream>
#include <time.h>

SCNImage::SCNImage()
{
    this->height = 0;
    this->width = 0;
    this->m_matrix = NULL;
}

SCNImage::SCNImage(const Mat &cv_img)
{
    this->height = cv_img.rows;
    this->width = cv_img.cols;
    this->m_matrix = new SCNPixelPtr[this->height * this->width];

    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }

        //std::cout << "(" << i << " " << j << ")" << std::endl;

        Vec3b intensity = cv_img.at<Vec3b>(i, j);
        uchar blue = intensity.val[0];
        uchar green = intensity.val[1];
        uchar red = intensity.val[2];
        this->m_matrix[i * this->width + j] = new SCNPixel(red, green, blue);
    }
}

SCNImage::SCNImage(int h, int w)
{
    this->height = h;
    this->width = w;
    this->m_matrix = new SCNPixelPtr[this->height * this->width];

    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }
        
        this->m_matrix[i * this->width + j] = new SCNPixel();
    }
}

SCNImage::SCNImage(const SCNImage & other)
{
    this->height = other.height;
    this->width = other.width;

    this->m_matrix = new SCNPixelPtr[this->height * this->width];

    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }
        
        this->m_matrix[i * this->width + j] = new SCNPixel(*other.m_matrix[i * this->width + j]);
    }
}

SCNImage::~SCNImage()
{
    this->destroyMe();
}

SCNImage & SCNImage::operator=(const SCNImage & other)
{
    this->height = other.height;
    this->width = other.width;
    this->m_matrix = new SCNPixelPtr[this->height * this->width];

    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }
        
        this->m_matrix[i * this->width + j] = new SCNPixel(*other.m_matrix[i * this->width + j]);
    }

    return *this;
}

void SCNImage::reverseColor()
{
    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }
        uchar r = this->m_matrix[i * this->width + j]->getR();
        uchar g = this->m_matrix[i * this->width + j]->getG();
        uchar b = this->m_matrix[i * this->width + j]->getB();
        delete this->m_matrix[i * this->width + j];

        this->m_matrix[i * this->width + j] = new SCNPixel(255 - r, 255 - g, 255 - b);
    }
}


void SCNImage::histogramEqualization()
{
    class hg
    {
    public:
        int accumulative;
        std::vector<SCNPixelPtr> pixels;
    public:
        hg():accumulative(0)
        {
        }
    };

    hg r_hg_list[256];
    hg g_hg_list[256];
    hg b_hg_list[256];

    std::cout << "************************* Histogram Equalization ************************" << std::endl;

    for (int i = -1, j = 0;; j++)
    {
        j %= this->width;
        if (j == 0)
        {
            i++;
        }
        if (i == this->height)
        {
            break;
        }
        uchar r = this->m_matrix[i * this->width + j]->getR();
        uchar g = this->m_matrix[i * this->width + j]->getG();
        uchar b = this->m_matrix[i * this->width + j]->getB();

        r_hg_list[r].accumulative++;
        r_hg_list[r].pixels.push_back(this->m_matrix[i * this->width + j]);

        g_hg_list[g].accumulative++;
        g_hg_list[g].pixels.push_back(this->m_matrix[i * this->width + j]);

        b_hg_list[b].accumulative++;
        b_hg_list[b].pixels.push_back(this->m_matrix[i * this->width + j]);
    }

    for (int i = 1; i < 256; i++)
    {
        r_hg_list[i].accumulative += r_hg_list[i - 1].accumulative;
        g_hg_list[i].accumulative += g_hg_list[i - 1].accumulative;
        b_hg_list[i].accumulative += b_hg_list[i - 1].accumulative;
    }

    int max_r_acc = r_hg_list[255].accumulative;
    int min_r_acc = r_hg_list[0].accumulative;
    double r_depth = (max_r_acc - min_r_acc) > 0 ? (max_r_acc - min_r_acc) : 1;

    int max_g_acc = g_hg_list[255].accumulative;
    int min_g_acc = g_hg_list[0].accumulative;
    double g_depth = (max_g_acc - min_g_acc) > 0 ? (max_g_acc - min_g_acc) : 1;

    int max_b_acc = b_hg_list[255].accumulative;
    int min_b_acc = b_hg_list[0].accumulative;
    double b_depth = (max_b_acc - min_b_acc) > 0 ? (max_b_acc - min_b_acc) : 1;


    for (int i = 0; i < 256; i++)
    {
        for (SCNPixelPtr ptr : r_hg_list[i].pixels)
        {
            ptr->setR(((r_hg_list[i].accumulative - min_r_acc) / r_depth) * 255);
        }

        for (SCNPixelPtr ptr : g_hg_list[i].pixels)
        {
            ptr->setG(((g_hg_list[i].accumulative - min_g_acc) / g_depth) * 255);
        }

        for (SCNPixelPtr ptr : b_hg_list[i].pixels)
        {
            ptr->setB(((b_hg_list[i].accumulative - min_b_acc) / b_depth) * 255);
        }
    }
}

SCNPixel & SCNImage::pixelAt(int row, int col) const
{
    if (col >= this->width || col < 0 || row >= this->height || row < 0)
    {
        throw "Index out of range";
    }

    return *(this->m_matrix[row * this->width + col]);
}

void SCNImage::setPixelAt(int row, int col, uchar gray)
{
    if (col >= this->width || col < 0 || row >= this->height || row < 0)
    {
        throw "Index out of range";
    }

    delete this->m_matrix[row * this->width + col];
    this->m_matrix[row * this->width + col] = new SCNPixel(gray);
}

void SCNImage::setPixelAt(int row, int col, uchar r, uchar g, uchar b)
{
    if (col >= this->width || col < 0 || row >= this->height || row < 0)
    {
        throw "Index out of range";
    }

    delete this->m_matrix[row * this->width + col];
    this->m_matrix[row * this->width + col] = new SCNPixel(r, g, b);
}

void SCNImage::subImage(int start_row, int start_col, int h, int w, SCNImage &dest) const
{
    if (start_row < 0)
    {
        start_row += this->height;
    }

    if (start_col < 0)
    {
        start_col += this->width;
    }

    if (start_row >= this->height)
    {
        start_row -= this->height;
    }

    if (start_col >= this->width)
    {
        start_col -= this->width;
    }

    dest.destroyMe();
    dest.height = h;
    dest.width = w;
    dest.m_matrix = new SCNPixelPtr[h * w];

    for (int i = start_row; i < start_row + h; i++)
    {
        for (int j = start_col; j < start_col + w; j++)
        {
            int row = i;
            int col = j;

            if (row < 0)
            {
                row += this->height;
            }

            if (col < 0)
            {
                col += this->width;
            }

            if (row >= this->height)
            {
                row -= this->height;
            }

            if (col >= this->width)
            {
                col -= this->width;
            }

            dest.m_matrix[(i - start_row) * w - start_col + j] = new SCNPixel(*(this->m_matrix[row * this->width + col]));
        }
    }
}

void SCNImage::toCVImage(Mat & cv_img)
{
    Mat new_img(this->height, this->width, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            // get pixel
            Vec3b color = new_img.at<Vec3b>(Point(j, i));

            // ... do something to the color ....
            color.val[0] = this->pixelAt(i, j).getR();
            color.val[1] = this->pixelAt(i, j).getG();
            color.val[2] = this->pixelAt(i, j).getB();

            // set pixel
            new_img.at<Vec3b>(Point(j, i)) = color;
        }
    }

    cv_img = new_img;
}

double SCNImage::getGrayVariance() const
{
    double sum = 0.0;
    double mean = 0.0;

    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            sum += this->m_matrix[i * this->width + j]->getGray();
        }
    }

    mean = sum / (this->width * this->height);

    sum = 0.0;
    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            uchar gray_color = this->m_matrix[i * this->width + j]->getGray();
            double offset = mean - gray_color;

            sum += offset * offset;
        }
    }

    return sum / (this->width * this->height);
}

void SCNImage::getCombinedImages(const Mat &cv_img, int window_size, int pyramid_start_size, SCNImage *combined)
{
    Mat image1(cv_img), image2;
    std::stack<SCNImage*> img_stack;
    int win_size = window_size;

    while (image1.rows * image1.cols > pyramid_start_size * pyramid_start_size)
    {
        SCNImage *new_img_ptr = new SCNImage(image1);
        img_stack.push(new_img_ptr);

        pyrDown(image1, image2, Size(image1.cols / 2, image1.rows / 2));
        image1 = image2;
    }

    MatchWindow w0, w1, w2;
    SCNImage *img_ptr = img_stack.top();
    match(*img_ptr, win_size, w0, w1, w2);
    img_stack.pop();
    delete img_ptr;


    int start_row = w0.getRowPos();
    int start_col = w0.getColPos();
    int dest1_row_start = w1.getRowPos();
    int dest1_col_start = w1.getColPos();
    int dest2_row_start = w2.getRowPos();
    int dest2_col_start = w2.getColPos();

    while (img_stack.size() > 0)
    {
        start_row *= 2;
        start_col *= 2;

        dest1_row_start *= 2;
        dest1_col_start *= 2;
        dest2_row_start *= 2;
        dest2_col_start *= 2;

        win_size *= 3;
        win_size /= 2;
        if (win_size % 2 == 0)
        {
            win_size += 1;
        }

        img_ptr = img_stack.top();
        match(*img_ptr, win_size, start_row, start_col, dest1_row_start, dest1_col_start, w1);
        match(*img_ptr, win_size, start_row, start_col, dest2_row_start, dest2_col_start, w2);
        dest1_row_start = w1.getRowPos();
        dest1_col_start = w1.getColPos();
        dest2_row_start = w2.getRowPos();
        dest2_col_start = w2.getColPos();
        
        img_stack.pop();
        delete img_ptr;
    }

    int rgb_ids[3];
    int image_id = 0;

    std::cout << "================= Combining color channels ... ==================" << std::endl;
    for (int r = 0; r < 3; r++)
    {
        for (int g = 0; g < 3; g++)
        {
            for (int b = 0; b < 3; b++)
            {
                if (r != g && r != b && g != b)
                {
                    rgb_ids[0] = r;
                    rgb_ids[1] = g;
                    rgb_ids[2] = b;
                    // std::cout << r << " " << g << " " << b << " " << std::endl;
                    combineImage(rgb_ids, cv_img, start_row, start_col, dest1_row_start, dest1_col_start, dest2_row_start, dest2_col_start, combined[image_id]);
                    image_id++;
                }
            }
        }
    }

}

int min(int a, int b, int c)
{
    int mi = a;
    
    if (b < mi)
    {
        mi = b;
    }

    if (c < mi)
    {
        mi = c;
    }

    return mi;
}

int max(int a, int b, int c)
{
    int ma = a;

    if (b > ma)
    {
        ma = b;
    }

    if (c > ma)
    {
        ma = c;
    }

    return ma;
}

int dist(int a, int b, int c)
{
    int ma = max(a, b, c);
    int mi = min(a, b, c);

    return (ma - mi) / 2;
}

void SCNImage::combineImage(int rgb_ids[3], const Mat &cv_img, int start_row, int start_col, int dest1_row_start, int dest1_col_start, int dest2_row_start, int dest2_col_start, SCNImage &combined)
{
    std::cout << "********************************************" << std::endl;

    int min_row = min(start_row, dest1_row_start, dest2_row_start);
    int min_col = min(start_col, dest1_col_start, dest2_col_start);

    int row_dist = dist(start_row, dest1_row_start, dest2_row_start);
    int col_dist = dist(start_col, dest1_col_start, dest2_col_start);

    if (row_dist < 50)
    {
        row_dist = cv_img.rows;
    }

    if (col_dist < 50)
    {
        col_dist = cv_img.cols;
    }

    int go_left = min_col;
    int go_up = min_row;

    std::cout << "Combined image size: " << row_dist << " " << col_dist << std::endl;
    SCNImage my_combined(row_dist, col_dist);
    //SCNImage origin(cv_img);
    uchar rgb[3];

    
    for (int i = -1, j = 0;; j++)
    {
        j %= col_dist;
        if (j == 0)
        {
            i++;
        }
        if (i == row_dist)
        {
            break;
        }
        int img0_row = start_row - go_up + i;
        int img0_col = start_col - go_left + j;
        if (img0_row < 0
            || img0_row >= cv_img.rows
            || img0_col < 0
            || img0_col >= cv_img.cols)
        {
            continue;
        }
        else
        {

            Vec3b intensity = cv_img.at<Vec3b>(img0_row, img0_col);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];

            rgb[0] = (blue + green + red) / 3;


            //rgb[0] = origin.pixelAt(img0_row, img0_col).getGray();
        }

        int img1_row = dest1_row_start - go_up + i;
        int img1_col = dest1_col_start - go_left + j;
        if (img1_row < 0
            || img1_row >= cv_img.rows
            || img1_col < 0
            || img1_col >= cv_img.cols)
        {
            continue;
        }
        else
        {

            Vec3b intensity = cv_img.at<Vec3b>(img1_row, img1_col);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];

            rgb[1] = (blue + green + red) / 3;


            //rgb[1] = origin.pixelAt(img1_row, img1_col).getGray();
        }

        int img2_row = dest2_row_start - go_up + i;
        int img2_col = dest2_col_start - go_left + j;
        if (img2_row < 0
            || img2_row >= cv_img.rows
            || img2_col < 0
            || img2_col >= cv_img.cols)
        {
            continue;
        }
        else
        {

            Vec3b intensity = cv_img.at<Vec3b>(img2_row, img2_col);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];

            rgb[2] = (blue + green + red) / 3;


            //rgb[2] = origin.pixelAt(img2_row, img2_col).getGray();
        }

        my_combined.setPixelAt(i, j, rgb[rgb_ids[0]], rgb[rgb_ids[1]], rgb[rgb_ids[2]]);
    }

    combined = my_combined;
}


void SCNImage::randomInitialMatchPoint(const SCNImage & img, int size, int & row, int & col)
{
    if (0 == size % 2)
    {
        throw "Size of match window should be odd!";
    }

    if (img.height < size || img.width < size)
    {
        throw "Size of match window is too large.";
    }

    /* initialize random seed: */
    srand(time(NULL));

    int rand_row = rand() % (img.height - size + 1) + size / 2;
    int rand_col = rand() % (img.width - size + 1) + size / 2;

    row = rand_row;
    col = rand_col;
}

class MatchWindowComparison
{
    bool reverse;

public:

    MatchWindowComparison(const bool& revparam = false)
    {
        reverse = revparam;
    }

    bool operator() (const MatchWindow &lhs, const MatchWindow &rhs) const
    {
        if (reverse) return (lhs.getKey() < rhs.getKey());
        else return (lhs.getKey() > rhs.getKey());
    }
};

void SCNImage::match(const SCNImage &img, int size, MatchWindow& win0, MatchWindow& win1, MatchWindow& win2)
{
    class VarianceComparison
    {
        bool reverse;

    public:

        VarianceComparison(const bool& revparam = false)
        {
            reverse = revparam;
        }

        bool operator() (const MatchWindow &lhs, const MatchWindow &rhs) const
        {
            if (reverse) return (lhs.getKey() > rhs.getKey());
            else return (lhs.getKey() < rhs.getKey());
        }
    };

    std::cout << "==================== Initial Template Match ======================" << std::endl;
    std::cout << "Rows: " << img.height << "\tCols: " << img.width << std::endl;
    std::cout << "Window Size: " << size << std::endl;

    // Get a sub image of the largest variance
    int row, col;
    int image_size = img.height > img.width ? img.height : img.width;
    std::priority_queue<MatchWindow, std::vector<MatchWindow>, VarianceComparison> variance_queue;

    for (int i = 0; i < 100; i++)
    {
        int r, c;
        randomInitialMatchPoint(img, size, r, c);
        SubImgBlock sib(r - size / 2, c - size / 2, size, size, img.height, img.width, img.m_matrix);
        variance_queue.push(MatchWindow(r, c, sib.getGrayVariance()));
    }

    row = variance_queue.top().getRowPos();
    col = variance_queue.top().getColPos();
    SubImgBlock sib1(row - size / 2, col - size / 2, size, size, img.height, img.width, img.m_matrix);
    std::cout << "Variance of sub1: " << sib1.getGrayVariance() << std::endl;

    // Search for the other two sub images that match sub1
    std::priority_queue<MatchWindow, std::vector<MatchWindow>, MatchWindowComparison> queue;

    win0 = MatchWindow(row, col, 0);

    /*
    select roughly two other places
    */

    for (int i = 0; i < img.height; i += size / 5)
    {
        for (int j = 0; j < img.width; j += size / 5)
        {
            if (i != row || j != col)
            {
                SubImgBlock sib2(i - size / 2, j - size / 2, size, size, img.height, img.width, img.m_matrix);
                double dist = sib1.imageDistance(sib2);
                queue.push(MatchWindow(i, j, dist));

                //std::cout << "Row: " << i << "\tCol: " << j << std::endl;
            }          
        }
    }

    std::cout << "Row: " << row << "\tCol: " << col << std::endl;

    int index = 0;
    while (queue.size() > 0)
    {
        if (2 == index)
        {
            break;
        }

        MatchWindow w = queue.top();
        queue.pop();

        if (index == 0)
        {
            // If the matched sub image is too close to the original sub image
            // Ignore it!
            int y_dist = w.getRowPos() - row;
            int x_dist = w.getColPos() - col;
            if (y_dist <= image_size / 4 && y_dist >= -image_size / 4)
            {
                if (x_dist <= image_size / 4 && x_dist >= -image_size / 4)
                {
                    continue;
                }
            }

            win1 = w;
            index++;
        }
        else if (index == 1)
        {
            // If the matched sub image is too close to the original sub image
            // Ignore it!
            int y_dist = w.getRowPos() - row;
            int x_dist = w.getColPos() - col;
            if (y_dist <= image_size / 4 && y_dist >= -image_size / 4)
            {
                if (x_dist <= image_size / 4 && x_dist >= -image_size / 4)
                {
                    continue;
                }
            }

            // If the matched sub image is too close to the previous sub image
            // Ignore it!
            y_dist = w.getRowPos() - win1.getRowPos();
            x_dist = w.getColPos() - win1.getColPos();
            if (y_dist <= image_size / 4 && y_dist >= -image_size / 4)
            {
                if (x_dist <= image_size / 4 && x_dist >= -image_size / 4)
                {
                    continue;
                }
            }

            win2 = w;
            index++;
        }
    }

    /*
    Narrower to accurate places (win1)
    */
    std::priority_queue<MatchWindow, std::vector<MatchWindow>, MatchWindowComparison> queue_1;
    int i_start = win1.getRowPos() - size / 5;
    int j_start = win1.getColPos() - size / 5;
    int i_end = win1.getRowPos() + size / 5 + 1;
    int j_end = win1.getColPos() + size / 5 + 1;

    for (int i = i_start; i < i_end; i++)
    {
        for (int j = j_start; j < j_end; j++)
        {
             SubImgBlock sib2(i - size / 2, j - size / 2, size, size, img.height, img.width, img.m_matrix);
             double dist = sib1.imageDistance(sib2);
             queue_1.push(MatchWindow(i, j, dist));
        }
    }

    if (queue_1.size() > 0)
    {
        win1 = queue_1.top();
        std::cout << "Row: " << win1.getRowPos() << "\tCol: " << win1.getColPos() << "\tDistance: " << win1.getKey() << std::endl;
    }

    /*
    Narrower to accurate places (win2)
    */
    std::priority_queue<MatchWindow, std::vector<MatchWindow>, MatchWindowComparison> queue_2;
    i_start = win2.getRowPos() - size / 5;
    j_start = win2.getColPos() - size / 5;
    i_end = win2.getRowPos() + size / 5 + 1;
    j_end = win2.getColPos() + size / 5 + 1;

    for (int i = i_start; i < i_end; i++)
    {
        for (int j = j_start; j < j_end; j++)
        {
            SubImgBlock sib2(i - size / 2, j - size / 2, size, size, img.height, img.width, img.m_matrix);
            double dist = sib1.imageDistance(sib2);
            queue_2.push(MatchWindow(i, j, dist));
        }
    }

    if (queue_2.size() > 0)
    {
        win2 = queue_2.top();
        std::cout << "Row: " << win2.getRowPos() << "\tCol: " << win2.getColPos() << "\tDistance: " << win2.getKey() << std::endl;
    }
}

void SCNImage::match(const SCNImage & img, int size, int row, int col, int dest_row_start, int dest_col_start, MatchWindow & dest)
{
    if (0 == size % 2)
    {
        throw "Size of match window should be odd!";
    }

    if (img.height < size || img.width < size)
    {
        throw "Size of match window is too large.";
    }

    int start_row = row;
    int start_col = col;

    std::cout << "======================== Image Pyramid Alignment =========================" << std::endl;
    std::cout << "Rows: " << img.height << "\tCols: " << img.width << std::endl;
    std::cout << "Window Size: " << size << std::endl;

    SubImgBlock sib1(start_row - size / 2, start_col - size / 2, size, size, img.height, img.width, img.m_matrix);
    std::priority_queue<MatchWindow, std::vector<MatchWindow>, MatchWindowComparison> queue;

    int i_start = dest_row_start - 2;
    int j_start = dest_col_start - 2;
    int i_end = dest_row_start + 2 + 1;
    int j_end = dest_col_start + 2 + 1;

    // std::cout << "i_start: " << i_start << "\tj_end: " << j_start << std::endl;
    // std::cout << "i_end: " << i_end << "\tj_end: " << j_end << std::endl;

    for (int i = i_start; i < i_end; i++)
    {
        for (int j = j_start; j < j_end; j++)
        {
            SubImgBlock sib2(i - size / 2, j - size / 2, size, size, img.height, img.width, img.m_matrix);
            double dist = sib1.imageDistance(sib2);
            queue.push(MatchWindow(i, j, dist));

            //std::cout << "Row: " << i << "\tCol: " << j << std::endl;
        }
    }

    std::cout << "Row: " << start_row << "\tCol: " << start_col << std::endl;

    if (queue.size() > 0)
    {
        MatchWindow w = queue.top();
        dest = w;
        std::cout << "Row: " << w.getRowPos() << "\tCol: " << w.getColPos() << "\tDistance: " << w.getKey() << std::endl;
    }
}


void SCNImage::destroyMe()
{
    if (NULL == this->m_matrix)
    {
        return;
    }
    // this->m_matrix is an array of pointers
    // Each pointer points to an instance of a pixel
    // We assume that each pixel is on the heap
    // So we should delete them
    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            delete this->m_matrix[i * width + j];
        }
    }

    // Delete the array of pointers
    delete this->m_matrix;

    this->height = 0;
    this->width = 0;
    this->m_matrix = NULL;
}


double SCNImage::imageDistance(const SCNImage & other)
{
    if (other.height != this->height || other.width != this->width)
    {
        throw "Sizes of these images are different!";
    }

    double sum = 0.0;
    double my_mean = 0.0;
    double your_mean = 0.0;

    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            sum += this->m_matrix[i * this->width + j]->getGray();
        }
    }

    my_mean = sum / (this->width * this->height);

    sum = 0.0;
    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            sum += other.m_matrix[i * this->width + j]->getGray();
        }
    }

    your_mean = sum / (this->width * this->height);

    double mean_offset = your_mean - my_mean;

    sum = 0.0;
    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            uchar left = this->m_matrix[i * this->width + j]->getGray();
            uchar right = other.m_matrix[i * this->width + j]->getGray();
            double offset = left + mean_offset - right;

            sum += offset * offset;
        }
    }

    return sum / (this->width * this->height);
}

bool SCNImage::overlap(int xs1[], int ys1[], int xs2[], int ys2[])
{
    for (int i = 0; i < 4; i++)
    {
        int x = xs1[i];
        int y = ys1[i];

        if (x >= xs2[0] && x <= xs2[2])
        {
            if (y >= ys2[0] && y <= ys2[2])
            {
                return true;
            }
        }
    }

    return false;
}
