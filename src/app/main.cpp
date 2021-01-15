#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "FaceImage.h"


static constexpr char  DATA_SET_PATH[] = R"(C:\)";
static constexpr float TRAIN_SET_RATIO = .7f;


std::vector<FaceImage> loadDataSet(std::string_view const& datasetDirectoryPath)
{
    for (auto&& entry : std::filesystem::directory_iterator(datasetDirectoryPath))
    {
        if (!entry.is_regular_file())
            continue;

        std::cout << entry.path().u8string() << '\n';
    }

    return {};
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
