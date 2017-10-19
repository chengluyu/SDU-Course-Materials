#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cout << "Usage: showimage [image path]" << std::endl;
        return 1;
    }
    // get the path of image file from application arguments
    cv::Mat image = cv::imread(argv[1], 1);
    cv::namedWindow("Image");
    cv::imshow("Image", image);
    // wait for user's key input
    cv::waitKey(0);
    return 0;
}
