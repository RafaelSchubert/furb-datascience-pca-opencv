#include <filesystem>
#include <list>
#include <regex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "FaceImage.h"


static constexpr char  DATA_SET_PATH[] = R"(C:\Users\Rafael\Pictures\ORL\)";
static constexpr float TRAIN_SET_RATIO = .7f;


std::list<std::filesystem::path> getDataSetFilesPaths(std::filesystem::path const& dataSetDirectoryPath)
{
    static std::regex const reExpectedFileExtensions{
            R"(^\d+_\d+\.j(?:p(?:eg|e|g)|fif?)$)",
            std::regex_constants::ECMAScript | std::regex_constants::icase
        };

    std::list<std::filesystem::path> dataSetFilesPaths;

    for (auto&& entry : std::filesystem::directory_iterator(dataSetDirectoryPath))
    {
        if (!entry.is_regular_file())
            continue;

        auto&& entryPath = entry.path();

        if (!std::regex_match(
                entryPath.filename().string(),
                reExpectedFileExtensions
            ))
        {
            continue;
        }

        dataSetFilesPaths.push_back(entryPath);
    }

    return dataSetFilesPaths;
}


FaceImage getFaceImageData(std::filesystem::path const& imageFilePath)
{
    static std::regex const reFileNameFormat{ R"(^(\d+)_(\d+)\.)" };

    auto const  fileName = imageFilePath.filename().string();
    std::smatch fileNameMatching;

    if (!std::regex_search(
            fileName,
            fileNameMatching,
            reFileNameFormat
        ))
    {
        return {};
    }

    auto imageData = cv::imread(
            imageFilePath.string(),
            cv::ImreadModes::IMREAD_GRAYSCALE
        );

    if (imageData.empty())
        return {};

    return FaceImage{
            std::move(imageData),
            std::stoul(fileNameMatching[1]),
            std::stoul(fileNameMatching[2])
        };
}


std::vector<FaceImage> loadDataSet(std::filesystem::path const& dataSetDirectoryPath)
{
    auto&& dataSetFilesPaths = getDataSetFilesPaths(dataSetDirectoryPath);

    if (empty(dataSetFilesPaths))
        return {};

    std::vector<FaceImage> dataSetEntries{ size(dataSetFilesPaths) };

    std::transform(
            begin(dataSetFilesPaths),
            end(dataSetFilesPaths),
            begin(dataSetEntries),
            getFaceImageData
        );

    return dataSetEntries;
}


std::pair<
        std::vector<FaceImage*>,
        std::vector<FaceImage*>
    > splitDataSet(
        std::vector<FaceImage> const& dataSet,
        float const                   trainRatio
    )
{
    return {};
}


int main()
{
    auto&& facesDataSet = loadDataSet(DATA_SET_PATH);

    auto&& [train, test] = splitDataSet(facesDataSet, TRAIN_SET_RATIO);

    return 0;
}
