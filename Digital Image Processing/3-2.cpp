#include <algorithm>
#include <climits>
#include <random>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class DisjointSet {
    std::vector<int> pre_, rank_;
public:
    DisjointSet(int size) {
        pre_.resize(size, 0);
        for (int i = 0; i < size; i++) {
            pre_[i] = i;
        }
        rank_.resize(size, 0);
    }
    
    int find(int x) {
        return pre_[x] == x ? x : (pre_[x] = find(pre_[x]));
    }
    
    void merge(int x, int y) {
        x = find(x); y = find(y);
        if (rank_[x] < rank_[y])
            std::swap(x, y);
        if (rank_[x] == rank_[y])
            rank_[x]++;
        pre_[y] = x;
    }
};


void fastConnectDomain(const cv::Mat & input, cv::Mat & output) {
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    DisjointSet set { input.rows * input.cols };
    // the first row
    for (int x = 1; x < input.cols; x++) {
        if (input.at<uchar>(0, x) == input.at<uchar>(0, x - 1)) {
            set.merge(x, x - 1);
        }
    }
    // the rest rows
    for (int y = 1; y < input.rows; y++) {
        // the first element of this row
        if (input.at<uchar>(y, 0) == input.at<uchar>(y - 1, 0)) {
            set.merge(y * input.cols, (y - 1) * input.cols);
        }
        // the rest elements of this row
        for (int x = 1; x < input.cols; x++) {
            int klass = y * input.cols + x;
            if (input.at<uchar>(y, x) == input.at<uchar>(y, x - 1)) {
                set.merge(klass, klass - 1);
            }
            if (input.at<uchar>(y, x) == input.at<uchar>(y - 1, x)) {
                set.merge(klass, klass - input.cols);
            }
            if (input.at<uchar>(y, x) == input.at<uchar>(y - 1, x - 1)) {
                set.merge(klass, klass - input.cols - 1);
            }
            if (input.at<uchar>(y - 1, x) == input.at<uchar>(y, x - 1)) {
                set.merge(klass - 1, klass - input.cols);
            }
        }
    }
    // klasses holds the mapping between class indices and colors
    std::unordered_map<int, cv::Vec3b> klasses;
    // uniform distributed random number generator
    std::random_device random_dev;
    std::mt19937_64 engine(random_dev());
    std::uniform_int_distribution<uchar> uniform_dist(0, UCHAR_MAX);
    // for all pixels
    for (int y = 0, i = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++, i++) {
            auto result = klasses.find(set.find(i));
            if (result == klasses.end()) {
                // generate a random color
                cv::Vec3b color {
                    uniform_dist(engine),
                    uniform_dist(engine),
                    uniform_dist(engine)
                };
                // dye this class with the color
                klasses[set[i]] = color;
                output.at<cv::Vec3b>(y, x) = color;
            } else {
                output.at<cv::Vec3b>(y, x) = result->second;
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat image = cv::imread("cc_input.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("Source Image", image);
    cv::Mat result;
    fastConnectDomain(image, result);
    cv::imshow("Apply Fast Connect Domain", result);
    cv::waitKey(0);
    return 0;
}

