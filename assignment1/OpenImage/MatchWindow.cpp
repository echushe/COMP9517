#include "MatchWindow.h"


MatchWindow::MatchWindow()
{
    this->m_dist = 0;
    this->m_row = 0;
    this->m_col = 0;
}

MatchWindow::MatchWindow(int row, int col, double key)
{
    this->m_dist = key;
    this->m_row = row;
    this->m_col = col;
}


MatchWindow::~MatchWindow()
{
}
