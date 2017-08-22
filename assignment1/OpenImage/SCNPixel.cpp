#include "SCNPixel.h"



SCNPixel::SCNPixel(uchar r, uchar g, uchar b)
{
    this->m_r = r;
    this->m_g = g;
    this->m_b = b;
}

SCNPixel::SCNPixel(uchar c)
{
    this->m_r = c;
    this->m_g = c;
    this->m_b = c;
}

SCNPixel::SCNPixel()
{
    this->m_r = 0;
    this->m_g = 0;
    this->m_b = 0;
}


SCNPixel::~SCNPixel()
{
}
