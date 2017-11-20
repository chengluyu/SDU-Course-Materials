#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


static void centerize(cv::Mat &spectrum) {
    spectrum = spectrum(cv::Rect { 0, 0, spectrum.cols & -2, spectrum.rows & -2 });
    int cx = spectrum.cols / 2;
    int cy = spectrum.rows / 2;
    cv::Mat q0 { spectrum, cv::Rect { 0, 0, cx, cy } };
    cv::Mat q1 { spectrum, cv::Rect { cx, 0, cx, cy } };
    cv::Mat q2 { spectrum, cv::Rect { 0, cy, cx, cy } };
    cv::Mat q3 { spectrum, cv::Rect { cx, cy, cx, cy } };
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

static cv::Mat optimizeForDFT(const cv::Mat &input) {
    int vsize = cv::getOptimalDFTSize(input.rows);
    int hsize = cv::getOptimalDFTSize(input.cols);
    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, vsize - input.rows, 0, hsize - input.cols, IPL_BORDER_CONSTANT, cv::Scalar::all(0));
    std::cout << "Original size: " << input.rows << 'x' << input.cols << '\n';
    std::cout << "Optimized size: " << padded.rows << 'x' << padded.cols << '\n';
    return padded;
}

int main(int argc, const char *argv[]) {
   cv::Mat input = cv::imread("mona-lisa.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat padded = optimizeForDFT(input);
    cv::Mat planes[2] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex, output;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    cv::split(complex, planes);
    cv::magnitude(planes[0], planes[1], output);
    output += cv::Scalar::all(1);
    cv::log(output, output);
    centerize(output);
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Origin", input);
    cv::imshow("Spectrum", output);
    cv::waitKey();
    return 0;
}