#pragma once

class MatchWindow
{
private:
    double m_dist;
    int m_row;
    int m_col;

public:
    MatchWindow();
    MatchWindow(int row, int col, double key);
    ~MatchWindow();

    double getKey() const
    {
        return this->m_dist;
    }

    int getRowPos() const
    {
        return this->m_row;
    }

    int getColPos() const
    {
        return this->m_col;
    }
};

