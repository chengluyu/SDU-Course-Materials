#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char * argv[]) {
    // check argv
    if (argc != 3) {
        std::cout << "Usage: this_program [foreground image] [background image]" << std::endl;
        return 1;
    }
    
    // load foreground and background image
    cv::Mat foreground = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat background = cv::imread(argv[2]);
    
    // check if the foreground image has alpha channel
    if (foreground.channels() == 3) {
        std::cerr << "Error: this image has no alpha channel" << std::endl;
        return 1;
    }
    
    // split the foreground image into four channels
    cv::Mat bgra[4];
    cv::split(foreground, bgra);
    
    // extract the color channels of foreground image
    cv::merge(bgra, 3, foreground);
    
    // extract alpha channel
    cv::Mat alpha;
    bgra[0] = bgra[1] = bgra[2] = bgra[3];
    cv::merge(bgra, 3, alpha);
    
    // show the image
    cv::imshow("Alpha channel", alpha);
    cv::waitKey(0);
    
    // resize background image to fit foreground image
    cv::resize(background, background, cv::Size {foreground.cols, foreground.rows});
    
    // convert to floating-point type because it will make multiplication easier
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    
    // do alpha blending: R = F * \alpha + B * (1 - \alpha)
    cv::Mat result = cv::Mat::zeros(foreground.size(), foreground.type());
    cv::multiply(alpha, foreground, foreground);
    cv::multiply(cv::Scalar::all(1.0) - alpha, background, background);
    cv::add(foreground, background, result);
    
    // show result image
    cv::imshow("Result of alpha blending", result / 255);
    cv::waitKey(0);
    return 0;
}