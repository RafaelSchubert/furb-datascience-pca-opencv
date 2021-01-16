#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"
#include "matrixOperations.h"


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    if (empty(trainSet))
        return;

    auto [eigenValues, eigenVectors] = eigenDecomposition(
            covarianceMatrix(
                    getDifferenceMatrix(
                            trainSet,
                            getMeanImage(trainSet)
                        )
                )
        );
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
