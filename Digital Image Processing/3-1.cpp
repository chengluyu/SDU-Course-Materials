#include <algorithm>
#include <array>
#include <climits>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void histogramEqualization(const cv::Mat & input, cv::Mat & output) {
    cv::Mat * channels = new cv::Mat [input.channels()];
    cv::split(input, channels);
    for (int i = 0; i < input.dims; i++) {
        std::array<int, UCHAR_MAX> h;
        h.fill(0);
        // counting PDF
        cv::Mat & channel = channels[i];
        for (int y = 0; y < channel.rows; y++) {
            for (int x = 0; x < channel.cols; x++) {
                int pixel = channel.at<uchar>(y, x);
                h[pixel]++;
            }
        }
        // compute CDF from PDF
        std::partial_sum(h.begin(), h.end(), h.begin());
        int cdf_min = *std::min_element(h.begin(), h.end());
        // equalization
        for (int i = 0; i < UCHAR_MAX; i++) {
            h[i] = (h[i] - cdf_min) * (UCHAR_MAX - 1) / ((channel.rows * channel.cols) - cdf_min);
        }
        // v = h(v)
        for (int y = 0; y < channel.rows; y++) {
            for (int x = 0; x < channel.cols; x++) {
                channel.at<uchar>(y, x) = h[channel.at<uchar>(y, x)];
            }
        }
    }
    cv::merge(channels, input.channels(), output);
    delete [] channels;
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("Lenna.png", cv::IMREAD_UNCHANGED);
    cv::imshow("Before Histogram Equalization", image);
    histogramEqualization(image, image);
    cv::imshow("After Histogram Equalization", image);
    cv::waitKey(0);
    return 0;
}
