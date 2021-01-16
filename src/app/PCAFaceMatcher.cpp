#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    if (empty(trainSet))
        return;

    auto differenceMatrix = getDifferenceMatrix(
            trainSet,
            getMeanImage(trainSet)
        );
}


cv::Mat PCAFaceMatcher::getMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& dataSet) const
{
    auto&& firstImage = dataSet.front().get().imageData;

    std::vector<unsigned long> pixelsTotals(
            static_cast<std::size_t>(firstImage.rows) * firstImage.cols,
            0UL
        );

    for (auto&& entryRef : dataSet)
    {
        auto&& entryImage = entryRef.get().imageData;

        std::transform(
                begin(pixelsTotals),
                end(pixelsTotals),
                entryImage.begin<unsigned char>(),
                begin(pixelsTotals),
                [](auto&& total, auto&& parcel) { return total + parcel; }
            );
    }

    cv::Mat meanImage = cv::Mat::zeros(
            firstImage.rows,
            firstImage.cols,
            CV_8UC1
        );

    std::transform(
            begin(pixelsTotals),
            end(pixelsTotals),
            meanImage.begin<unsigned char>(),
            [entryCount = size(dataSet)](auto&& total) { return static_cast<unsigned char>(total / entryCount); }
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
            size(dataSet),
            CV_16SC1
        );

    for (std::size_t idxEntry = 0; idxEntry < size(dataSet); ++idxEntry)
    {
        auto&& entryImageData = dataSet[idxEntry].get().imageData;

        for (int row = 0; row < entryImageData.rows; ++row)
        {
            differenceMatrix.at<short>(row, idxEntry) =
                static_cast<short>(entryImageData.at<unsigned char>(row, 0))
                - meanImage.at<unsigned char>(row, 0);
        }
    }

    return differenceMatrix;
}
