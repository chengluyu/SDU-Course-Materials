#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void gaussianFilter(const cv::Mat & input, cv::Mat & output, double sigma) {
    int radius = std::floor(6 * sigma + 1), size = (radius + 1) * (radius + 1);
    // convert input image to floating-point data type
    cv::Mat origin;
    input.convertTo(origin, CV_32FC3);
    // compute Gaussian function values
    double * vec = radius + new double [size];
    double sum = vec[0] = 1.0;
    for (int x = 1; x <= radius; x++) {
        double g = std::exp(- x * x / (2 * sigma * sigma));
        vec[-x] = vec[x] = g;
        sum += 2 * g;
    }
    // normalization
    for (int i = -radius; i <= radius; i++) {
        vec[i] /= sum;
    }
    // iteration on columns;
    cv::Mat middle = cv::Mat::zeros(input.rows, input.cols, CV_32FC3);
    for (int x = 0; x < origin.cols; x++) {
        cv::Mat sum = cv::Mat::zeros(origin.rows, 1, CV_32FC3);
        for (int offset = -radius; offset <= radius; offset++) {
            cv::Mat col;
            int src = x + offset;
            if (src < 0) {
                src = -src;
            } else if (src >= origin.cols) {
                src = origin.cols - (src - origin.cols) - 1;
            }
            cv::multiply(origin.col(src), cv::Scalar::all(vec[offset]), col);
            cv::add(sum, col, sum);
        }
        sum.copyTo(middle.col(x));
    }
    // iteration on rows
    cv::Mat result = cv::Mat::zeros(input.rows, input.cols, CV_32FC3);
    for (int y = 0; y < input.rows; y++) {
        cv::Mat sum = cv::Mat::zeros(1, middle.cols, CV_32FC3);
        for (int offset = -radius; offset <= radius; offset++) {
            int src = y + offset;
            if (src < 0)
                src = -src;
            else if (src >= middle.rows)
                src = middle.rows - (src - middle.rows) - 1;
            cv::Mat row;
            cv::multiply(middle.row(src), cv::Scalar::all(vec[offset]), row);
            cv::add(sum, row, sum);
        }
        sum.copyTo(result.row(y));
    }
    // transform to the original data type
    result.convertTo(output, input.type());
    delete [] (vec - radius);
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("Lenna.png", cv::IMREAD_UNCHANGED);
    cv::imshow("Original Image", image);
    gaussianFilter(image, image, 4.0);
    cv::imshow("After Applying Gaussian Filter", image);
    cv::waitKey(0);
    return 0;
}
