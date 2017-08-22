#pragma once
typedef unsigned char uchar;

class SCNPixel
{
private:
    uchar m_r;
    uchar m_g;
    uchar m_b;
public:
    SCNPixel(uchar r, uchar g, uchar b);
    SCNPixel(uchar c);
    SCNPixel();
    ~SCNPixel();

    uchar getR() const
    {
        return this->m_r;
    }

    uchar getG() const
    {
        return this->m_g;
    }

    uchar getB() const
    {
        return this->m_b;
    }

    void setR(uchar r)
    {
        this->m_r = r;
    }

    void setG(uchar g)
    {
        this->m_g = g;
    }

    void setB(uchar b)
    {
        this->m_b = b;
    }

    uchar getGray() const
    {
        return (this->m_r + this->m_g + this->m_b) / 3;
    }
};

