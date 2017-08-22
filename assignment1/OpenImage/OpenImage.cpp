#include "./RGBSolver.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    if (argc == 2)
    {
        RGBSolver solver(argv[1], 15, 40);
    }
    else if (argc == 3)
    {
        RGBSolver solver(argv[1], std::stoi(argv[2]), 40);
    }
    else if (argc == 4)
    {
        RGBSolver solver(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
    }
    
    
    return 0;
}