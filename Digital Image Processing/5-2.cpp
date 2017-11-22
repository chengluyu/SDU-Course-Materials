#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


static cv::Mat optimizeForDFT(const cv::Mat &input) {
    int vsize = cv::getOptimalDFTSize(input.rows);
    int hsize = cv::getOptimalDFTSize(input.cols);
    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, vsize - input.rows, 0, hsize - input.cols, IPL_BORDER_CONSTANT, cv::Scalar::all(0));
    return padded;
}

void removePixel(cv::Mat &image, int y, int x) {
    int rows = image.rows;
    int cols = image.cols;
    if (y < 0) y += rows;
    if (x < 0) x += cols;
    image.at<cv::Vec2d>(y, x) = { 0.0, 0.0 };
}

int main(int argc, const char * argv[]) {
    cv::Mat input = cv::imread("noise.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat optimizedInput = optimizeForDFT(input);
    optimizedInput.convertTo(optimizedInput, CV_64F);
    cv::Mat fourier;
    cv::dft(optimizedInput, fourier, cv::DFT_COMPLEX_OUTPUT);
    int pointsToRemove[8][2] = {
        { -1, 9 }, { -1, -9 }, { 0, 10 }, { 0, -10 },
        { 0, 9 }, { 0, -9 }, { 0, 8 }, { 0, -8 }
    };
    for (auto yx : pointsToRemove)
        removePixel(fourier, yx[0], yx[1]);
    cv::Mat output;
    cv::dft(fourier, output, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("before", input);
    cv::imshow("after", output);
    cv::waitKey();
    return 0;
}