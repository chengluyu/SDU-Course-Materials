#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>


template <typename Num>
inline Num sq(Num x) { return x * x; }

cv::Mat convolveDFT(const cv::Mat &image, const cv::Mat &stencil) {
    cv::Mat im;
    cv::Mat tpl;
    cv::Mat result;
    image.convertTo(im, CV_64F);
    stencil.convertTo(tpl, CV_64F);
    cv::flip(tpl, tpl, -1);
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(im.cols);
    dftSize.height = cv::getOptimalDFTSize(im.rows);
    cv::Mat tempA(dftSize, im.type(), cv::Scalar::all(0));
    cv::Mat tempB(dftSize, tpl.type(), cv::Scalar::all(0));
    cv::Mat roiA(tempA, cv::Rect(0, 0, im.cols, im.rows));
    im.copyTo(roiA);
    cv::Mat roiB(tempB, cv::Rect(0, 0, tpl.cols, tpl.rows));
    tpl.copyTo(roiB);
    cv::dft(tempA, tempA, 0, im.rows);
    cv::dft(tempB, tempB, 0, tpl.rows);
    cv::mulSpectrums(tempA, tempB, tempA, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(tempA, tempA, cv::DFT_INVERSE | cv::DFT_SCALE, im.rows);
    tempA(cv::Rect(0, 0, im.cols, im.rows)).copyTo(result);
    return result;
}

cv::Mat multipleChannelConvolve(const cv::Mat &im, const cv::Mat &tpl) {
    cv::Mat ims[3];
    cv::Mat tpls[3];
    cv::split(im, ims);
    cv::split(tpl, tpls);
    cv::Mat conv[3];
    conv[0] = convolveDFT(ims[0], tpls[0]);
    conv[1] = convolveDFT(ims[1], tpls[1]);
    conv[2] = convolveDFT(ims[2], tpls[2]);
    cv::Mat result;
    cv::merge(conv, 3, result);
    return result;
}

cv::Rect fastTemplateMatch(const cv::Mat &image, const cv::Mat stencil) {
    auto **sqsum = new int64_t*[image.rows + 1];
    for (int y = 0; y < image.rows + 1; y++)
        sqsum[y] = new int64_t[image.cols + 1];
    std::fill(sqsum[0], sqsum[0] + image.cols + 1, 0);
    for (int y = 1; y < image.rows + 1; y++) {
        int64_t row_sum = sqsum[y][0] = 0;
        for (int x = 1; x < image.cols + 1; x++) {
            cv::Vec3b vec = image.at<cv::Vec3b>(y - 1, x - 1);
            row_sum += sq<int64_t>(vec[0]) + sq<int64_t>(vec[1]) + sq<int64_t>(vec[2]);
            sqsum[y][x] = row_sum + sqsum[y - 1][x];
        }
    }
    auto third_term = [&](int y, int x) {
        return sqsum[y + 1][x + 1]
               - sqsum[y - stencil.rows][x + 1]
               - sqsum[y + 1][x - stencil.cols]
               + sqsum[y - stencil.rows][x - stencil.cols];
    };
    cv::Mat convolved = multipleChannelConvolve(image, stencil);
    double min_diff = std::numeric_limits<double>::max();
    int min_y = 0;
    int min_x = 0;
    for (int y = stencil.rows; y < image.rows; y++) {
        for (int x = stencil.cols; x < image.cols; x++) {
            cv::Vec3d u = convolved.at<cv::Vec3d>(y, x);
            double diff = - 2 * (u[0] + u[1] + u[2]) + static_cast<double>(third_term(y, x));
            if (diff < min_diff) {
                min_diff = diff;
                min_y = y;
                min_x = x;
            }
        }
    }
    for (int y = 0; y < image.rows; y++)
        delete[] sqsum[y];
    delete[] sqsum;
    std::clog << "minimal difference is " << min_diff << '\n';
    std::clog << "x = " << min_x - stencil.cols << ", y = " << min_y - stencil.rows << '\n';
    return cv::Rect{ min_x - stencil.cols, min_y - stencil.rows, stencil.cols, stencil.rows };
}


int main() {
    cv::Mat image = cv::imread("mona-lisa.png", cv::IMREAD_COLOR);
    cv::Mat stencil = cv::imread("template.png", cv::IMREAD_COLOR);
    cv::Rect region = fastTemplateMatch(image, stencil);
    cv::rectangle(image, region, cv::Scalar{ 0, 0, 255 }, 2);
    cv::imshow("Matching Result", image);
    cv::waitKey();
    return 0;
}