#include <cassert>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


void meanFilter(const cv::Mat & input, cv::Mat & output, int radius) {
    assert(input.type() == CV_8UC3);
    cv::Mat bordered;
    cv::copyMakeBorder(input, bordered, radius + 1, radius + 1, radius + 1, radius + 1, cv::BORDER_REFLECT);
    cv::Mat integral = cv::Mat::zeros(bordered.rows, bordered.cols, CV_32SC3);
    cv::Vec3i sum { 0, 0, 0 };
    for (int x = 0; x < bordered.cols; x++)
        integral.at<cv::Vec3i>(0, x) = (sum += bordered.at<cv::Vec3b>(0, x));
    for (int y = 1; y < bordered.rows; y++) {
        sum = { 0, 0, 0 };
        for (int x = 0; x < bordered.cols; x++)
            integral.at<cv::Vec3i>(y, x) = integral.at<cv::Vec3i>(y - 1, x) + (sum += bordered.at<cv::Vec3b>(y, x));
    }
    int diameter = 2 * radius + 1;
    int area = diameter * diameter;
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            int xx = x + diameter, yy = y + diameter;
            output.at<cv::Vec3b>(y, x) = (integral.at<cv::Vec3i>(yy, xx) - integral.at<cv::Vec3i>(y, xx) - integral.at<cv::Vec3i>(yy, x) + integral.at<cv::Vec3i>(y, x)) / area;
        }
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("Lenna.png", cv::IMREAD_UNCHANGED);
    cv::imshow("Original Image", image);
    meanFilter(image, image, 2);
    cv::imshow("After Applying Mean Filter", image);
    cv::waitKey(0);
    return 0;
}
