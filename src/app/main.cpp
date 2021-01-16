#include <filesystem>
#include <list>
#include <map>
#include <regex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "FaceImage.h"


static cv::Size const  IMAGE_SIZE{ 80, 80 };
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


cv::Mat getEntryImageData(std::filesystem::path const& entryFilePath)
{
    cv::Mat resizedImageData;

    {
        auto&& imageData = cv::imread(
                entryFilePath.string(),
                cv::ImreadModes::IMREAD_GRAYSCALE
            );

        cv::resize(
                imageData,
                resizedImageData,
                IMAGE_SIZE
            );
    }

    return cv::Mat{ resizedImageData.t() }
        .reshape(
                1,
                IMAGE_SIZE.width * IMAGE_SIZE.height
            );
}


FaceImage readDataSetEntry(std::filesystem::path const& entryFilePath)
{
    static std::regex const reFileNameFormat{ R"(^(\d+)_(\d+)\.)" };

    auto const  fileName = entryFilePath.filename().string();
    std::smatch fileNameMatching;

    if (!std::regex_search(
            fileName,
            fileNameMatching,
            reFileNameFormat
        ))
    {
        return {};
    }

    return FaceImage{
            getEntryImageData(entryFilePath),
            std::stoul(fileNameMatching[1]),
            std::stoul(fileNameMatching[2])
        };
}


std::list<FaceImage> loadDataSet(std::filesystem::path const& dataSetDirectoryPath)
{
    auto&& dataSetFilesPaths = getDataSetFilesPaths(dataSetDirectoryPath);

    if (empty(dataSetFilesPaths))
        return {};

    std::list<FaceImage> dataSetEntries;

    std::transform(
            begin(dataSetFilesPaths),
            end(dataSetFilesPaths),
            std::back_inserter(dataSetEntries),
            readDataSetEntry
        );

    return dataSetEntries;
}


std::pair<
        std::list<FaceImage*>,
        std::list<FaceImage*>
    > splitDataSet(
        std::list<FaceImage>& dataSet,
        float const           trainRatio
    )
{
    std::list<FaceImage*> trainDataSet;
    std::list<FaceImage*> testDataSet;

    {
        std::map<
                unsigned int,
                std::list<FaceImage*>
            > mapEntriesPerClass;

        for (auto&& entry : dataSet)
        {
            auto itClassEntriesPair = mapEntriesPerClass.find(entry.faceId);

            if (itClassEntriesPair == end(mapEntriesPerClass))
            {
                itClassEntriesPair = mapEntriesPerClass.insert(std::make_pair(
                        entry.faceId,
                        std::list<FaceImage*>{}
                    )).first;
            }

            itClassEntriesPair->second.push_back(&entry);
        }

        auto itTrainDataSetInsert = std::back_inserter(trainDataSet);
        auto itTestDataSetInsert  = std::back_inserter(testDataSet);

        for (auto&& [classId, classEntriesPtrs] : mapEntriesPerClass)
        {
            if (empty(classEntriesPtrs))
                continue;

            std::size_t trainEntriesCount = size(classEntriesPtrs) * trainRatio;
            auto        splitPoint        = begin(classEntriesPtrs);

            for (; trainEntriesCount; --trainEntriesCount)
                ++splitPoint;

            std::copy(
                    begin(classEntriesPtrs),
                    splitPoint,
                    itTrainDataSetInsert
                );

            std::copy(
                    splitPoint,
                    end(classEntriesPtrs),
                    itTestDataSetInsert
                );
        }
    }

    return {
            std::move(trainDataSet),
            std::move(testDataSet)
        };
}


int main()
{
    auto&& facesDataSet = loadDataSet(DATA_SET_PATH);

    auto&& [train, test] = splitDataSet(facesDataSet, TRAIN_SET_RATIO);

    return 0;
}
