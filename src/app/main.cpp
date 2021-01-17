#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "FaceImage.h"
#include "PCAFaceMatcher.h"


static cv::Size const                IMAGE_SIZE{ 80, 80 };
static constexpr std::pair<int, int> RANGE_OF_COMPONENTS{ 10, 20 };
static constexpr char                DATA_SET_PATH[] = R"(C:\Users\Rafael\Pictures\ORL\)";
static constexpr float               TRAIN_SET_RATIO = .7f;


std::list<std::filesystem::path> getDataSetFilesPaths(std::filesystem::path const& dataSetDirectoryPath)
{
    static std::regex const reExpectedFileExtensions(
            R"(^\d+_\d+\.j(?:p(?:eg|e|g)|fif?)$)",
            std::regex_constants::ECMAScript | std::regex_constants::icase
        );

    std::list<std::filesystem::path> dataSetFilesPaths;

    for (auto const& entry : std::filesystem::directory_iterator(dataSetDirectoryPath))
    {
        if (!entry.is_regular_file())
            continue;

        auto const& entryPath = entry.path();

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
    cv::Mat finalImageData;

    {
        cv::Mat resizedImageData;

        {
            auto imageData = cv::imread(
                    entryFilePath.string(),
                    cv::ImreadModes::IMREAD_GRAYSCALE
                );

            cv::resize(
                    imageData,
                    resizedImageData,
                    IMAGE_SIZE
                );
        }

        resizedImageData.convertTo(finalImageData, CV_64FC1);
    }

    return cv::Mat(finalImageData.t())
        .reshape(
                1,
                IMAGE_SIZE.width * IMAGE_SIZE.height
            );
}


FaceImage readDataSetEntry(std::filesystem::path const& entryFilePath)
{
    static std::regex const reFileNameFormat(R"(^(\d+)_(\d+)\.)");

    auto        fileName = entryFilePath.filename().string();
    std::smatch fileNameMatching;

    if (!std::regex_search(
            fileName,
            fileNameMatching,
            reFileNameFormat
        ))
    {
        return {};
    }

    return {
            getEntryImageData(entryFilePath),
            std::stoul(fileNameMatching[1]),
            std::stoul(fileNameMatching[2])
        };
}


std::vector<FaceImage> loadDataSet(std::filesystem::path const& dataSetDirectoryPath)
{
    auto dataSetFilesPaths = getDataSetFilesPaths(dataSetDirectoryPath);

    if (empty(dataSetFilesPaths))
        return {};

    std::vector<FaceImage> dataSetEntries(size(dataSetFilesPaths));

    std::transform(
            begin(dataSetFilesPaths),
            end(dataSetFilesPaths),
            begin(dataSetEntries),
            readDataSetEntry
        );

    return dataSetEntries;
}


std::pair<
        std::vector<std::reference_wrapper<FaceImage>>,
        std::vector<std::reference_wrapper<FaceImage>>
    > splitDataSet(
        std::vector<FaceImage>& dataSet,
        float const             trainRatio
    )
{
    std::vector<std::reference_wrapper<FaceImage>> trainSet;
    std::vector<std::reference_wrapper<FaceImage>> testSet;

    trainSet.reserve(size(dataSet));
    testSet .reserve(size(dataSet));

    {
        std::map<
                unsigned int,
                std::list<std::reference_wrapper<FaceImage>>
            > mapClassEntries;

        for (auto& entry : dataSet)
        {
            auto itClassEntriesPair = mapClassEntries.find(entry.faceId);

            if (itClassEntriesPair == end(mapClassEntries))
            {
                itClassEntriesPair = mapClassEntries.insert(std::make_pair(
                        entry.faceId,
                        std::list<std::reference_wrapper<FaceImage>>{}
                    )).first;
            }

            itClassEntriesPair->second.emplace_back(entry);
        }

        auto itTrainDataSetInsert = std::back_inserter(trainSet);
        auto itTestDataSetInsert  = std::back_inserter(testSet);

        std::random_device randDev;
        std::mt19937       numGenerator(randDev());

        for (auto const& [classId, classEntries] : mapClassEntries)
        {
            if (empty(classEntries))
                continue;

            std::vector<std::reference_wrapper<FaceImage>> shuffledClassEntries(
                    begin(classEntries),
                    end(classEntries)
                );

            std::shuffle(
                    begin(shuffledClassEntries),
                    end(shuffledClassEntries),
                    numGenerator
                );

            auto itSplitPoint = begin(shuffledClassEntries) + static_cast<std::size_t>(size(shuffledClassEntries) * trainRatio);

            std::copy(
                    begin(shuffledClassEntries),
                    itSplitPoint,
                    itTrainDataSetInsert
                );

            std::copy(
                    itSplitPoint,
                    end(shuffledClassEntries),
                    itTestDataSetInsert
                );
        }
    }

    return {
            std::move(trainSet),
            std::move(testSet)
        };
}


double scoreRecognition(
        std::vector<std::reference_wrapper<FaceImage>> const& trainSet,
        std::vector<std::reference_wrapper<FaceImage>> const& testSet,
        int const                                             numberOfComponents
    )
{
    PCAFaceMatcher matcher(numberOfComponents);

    matcher.train(trainSet);

    auto correctPredictions = std::count_if(
            begin(testSet),
            end(testSet),
            [&](auto const& entryRef)
            {
                auto const& entry = entryRef.get();

                auto [predictedClass, confidenceRate] = matcher.predict(entry);

                return entry.faceId == predictedClass;
            }
        );

    return static_cast<double>(correctPredictions) / size(testSet);
}


int main()
{
    auto facesDataSet = loadDataSet(DATA_SET_PATH);

    auto [trainSet, testSet] = splitDataSet(facesDataSet, TRAIN_SET_RATIO);

    for (
            int numberOfComponents = RANGE_OF_COMPONENTS.first;
            numberOfComponents <= RANGE_OF_COMPONENTS.second;
            ++numberOfComponents
        )
    {
        auto accuracyRate = scoreRecognition(
                trainSet,
                testSet,
                numberOfComponents
            );

        std::printf(
                "%i componentes principais, acuracia: %.2f%%.\n",
                numberOfComponents,
                100. * accuracyRate
            );
    }

    return 0;
}
