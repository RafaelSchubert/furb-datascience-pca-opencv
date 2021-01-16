#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    if (empty(trainSet))
        return;

    auto covariance = getCovarianceMatrix(
            getDifferenceMatrix(
                    trainSet,
                    getMeanImage(trainSet)
                )
        );
}


cv::Mat PCAFaceMatcher::multiplyMatrices(
        cv::Mat left,
        cv::Mat right
    )
{
    cv::Mat resultMatrix = cv::Mat::zeros(
            left.rows,
            right.cols,
            left.type()
        );

    for (int row = 0; row < left.rows; ++row)
    {
        for (int col = 0; col < right.cols; ++col)
        {
            for (int element = 0; element < left.cols; ++element)
                resultMatrix.at<double>(row, col) += left.at<double>(row, element) * right.at<double>(element, col);
        }
    }

    return resultMatrix;
}


cv::Mat PCAFaceMatcher::getMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& dataSet) const
{
    auto&& firstImage = dataSet.front().get().imageData;

    cv::Mat meanImage = cv::Mat::zeros(
            firstImage.rows,
            firstImage.cols,
            firstImage.type()
        );

    for (auto&& entryRef : dataSet)
    {
        auto&& entryImage = entryRef.get().imageData;

        std::transform(
                meanImage.begin<double>(),
                meanImage.end<double>(),
                entryImage.begin<double>(),
                meanImage.begin<double>(),
                [](auto&& total, auto&& parcel) { return total + parcel; }
            );
    }

    std::transform(
            meanImage.begin<double>(),
            meanImage.end<double>(),
            meanImage.begin<double>(),
            [entryCount = size(dataSet)](auto&& total) { return total / entryCount; }
        );

    return meanImage;
}


cv::Mat PCAFaceMatcher::getDifferenceMatrix(
        std::vector<std::reference_wrapper<FaceImage>> const& dataSet,
        cv::Mat const&                                        meanImage
    ) const
{
    auto&& firstImage = dataSet.front().get().imageData;

    cv::Mat differenceMatrix = cv::Mat::zeros(
            firstImage.rows,
            static_cast<int>(size(dataSet)),
            firstImage.type()
        );

    for (int col = 0; col < differenceMatrix.cols; ++col)
    {
        auto&& entryImageData = dataSet[col].get().imageData;

        for (int row = 0; row < differenceMatrix.rows; ++row)
            differenceMatrix.at<double>(row, col) = entryImageData.at<double>(row, 0) - meanImage.at<double>(row, 0);
    }

    return differenceMatrix;
}


cv::Mat PCAFaceMatcher::getCovarianceMatrix(cv::Mat const& matrix) const
{
    return PCAFaceMatcher::multiplyMatrices(
            matrix.t(),
            matrix
        );
}
