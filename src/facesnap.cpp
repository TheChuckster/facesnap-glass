#include "cpu_cv.h"
#include "cuda_test.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    // return openCVtest();
    return cuda_test();
}
