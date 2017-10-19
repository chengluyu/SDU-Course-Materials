#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

template <typename VectorType>
VectorType interpolate(const cv::Mat &src, double x, double y) {
    int x1 = std::floor(x), x2 = std::ceil(x);
    int y1 = std::floor(y), y2 = std::ceil(y);
    
    VectorType p, q;
    double dx = x2 - x1, dx1 = x - x1, dx2 = x2 - x;
    if (x1 == x2) {
        p = src.at<VectorType>(y1, x1);
        q = src.at<VectorType>(y2, x1);
    } else {
        VectorType a = src.at<VectorType>(y1, x1);
        VectorType b = src.at<VectorType>(y1, x2);
        VectorType c = src.at<VectorType>(y2, x1);
        VectorType d = src.at<VectorType>(y2, x2);
        p = dx1 / dx * a + dx2 / dx * b;
        q = dx1 / dx * c + dx2 / dx * d;
    }
    
    VectorType result;
    double dy = y2 - y1, dy1 = y - y1, dy2 = y2 - y;
    if (y1 == y2) {
        result = p;
    } else {
        result = (dy1 / dy) * p + (dy2 / dy) * q;
    }
    return result;
}

void transform(const cv::Mat &src, cv::Mat &dst) {
    double half_height = 0.5 * src.rows, half_width = 0.5 * src.cols;
    for (int iy = 0; iy < dst.rows; iy++) {
        for (int ix = 0; ix < dst.cols; ix++) {
            double y = (iy - half_height) / half_height;
            double x = (ix - half_width) / half_width;
            double r = std::sqrt(y * y + x * x);
            if (r < 1.0) {
                double theta = (1 - r) * (1 - r);
                double xx = std::cos(theta) * x - std::sin(theta) * y;
                double yy = std::sin(theta) * x + std::cos(theta) * y;
                yy = yy * half_height + half_height;
                xx = xx * half_width + half_width;
                dst.at<cv::Vec3b>(cv::Point(ix, iy)) = interpolate<cv::Vec3b>(src, xx, yy);
            } else {
                dst.at<cv::Vec3b>(cv::Point(ix, iy)) = src.at<cv::Vec3b>(cv::Point(ix, iy));
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("input.png");
    cv::Mat result = cv::Mat::zeros(image.rows, image.cols, image.type());
    transform(image, result);
    cv::imshow("Transform result", result);
    cv::waitKey(0);
    return 0;
}
