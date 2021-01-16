#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    if (empty(trainSet))
        return;

    auto meanImage = getMeanImage(trainSet);
}


cv::Mat PCAFaceMatcher::getMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& trainSet) const
{
    auto&& firstImage = trainSet[0].get().imageData;

    std::vector<unsigned long> pixelsTotals(
            static_cast<std::size_t>(firstImage.rows) * firstImage.cols,
            0UL
        );

    for (auto&& entryRef : trainSet)
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
            [entryCount = size(trainSet)](auto&& total) { return static_cast<unsigned char>(total / entryCount); }
        );

    return meanImage;
}
