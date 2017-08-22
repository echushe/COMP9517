#include <iostream>

struct Timer
{
    Timer()
    {

    }

    void printTest()
    {
        std::cout << "print test in timer!" << std::endl;
    }
};

struct TimeKeeper
{
    TimeKeeper(const Timer &t)
    {
    }

    void printTest()
    {
        std::cout << "print test in time keeper!" << std::endl;
    }
};

int main()
{
    Timer a;
    TimeKeeper keeper(a);
    keeper.printTest();
}