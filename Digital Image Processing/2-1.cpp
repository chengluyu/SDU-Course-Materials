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

void scale(cv::Mat &src, cv::Mat &dst, double sx, double sy) {
    src.convertTo(src, CV_32FC3, 1.0 / 255);
    dst = cv::Mat::zeros(static_cast<int>(src.rows * sy),
                         static_cast<int>(src.cols * sx),
                         src.type());
    for (int iy = 0; iy < dst.rows; iy++) {
        for (int ix = 0; ix < dst.cols; ix++) {
            double y = iy / sy, x = ix / sx;
            dst.at<cv::Vec3f>(iy, ix) = interpolate<cv::Vec3f>(src, x, y);
        }
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("input.png");
    cv::Mat result;
    scale(image, result, 0.3, 0.3);
    cv::imshow("Origin image", image);
    cv::imshow("Scaling result", result);
    cv::waitKey(0);
    return 0;
}
