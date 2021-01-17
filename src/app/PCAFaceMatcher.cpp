#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"
#include "matrixOperations.h"


PCAFaceMatcher::PCAFaceMatcher(int numberOfComponents) :
    m_numberOfComponents(numberOfComponents)
{ }


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    clear();

    if (empty(trainSet))
        return;

    calculateMeanImage(trainSet);

    auto const differenceMatrix = getDifferenceMatrix(
            trainSet,
            m_mean
        );

    calculateEigenFaces(differenceMatrix);
    calculateProjections(
            trainSet,
            differenceMatrix
        );
}


void PCAFaceMatcher::clear()
{
    m_mean        = {};
    m_eigenFaces  = {};
    m_projections = {};
    m_classes     = {};
}


void PCAFaceMatcher::calculateMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& dataSet)
{
    auto&& firstImage = dataSet.front().get().imageData;

    m_mean = cv::Mat::zeros(
            firstImage.rows,
            firstImage.cols,
            firstImage.type()
        );

    for (auto&& entryRef : dataSet)
    {
        auto&& entryImage = entryRef.get().imageData;

        std::transform(
                m_mean.begin<double>(),
                m_mean.end<double>(),
                entryImage.begin<double>(),
                m_mean.begin<double>(),
                [](auto&& total, auto&& parcel) { return total + parcel; }
            );
    }

    std::transform(
            m_mean.begin<double>(),
            m_mean.end<double>(),
            m_mean.begin<double>(),
            [entryCount = size(dataSet)](auto&& total) { return total / entryCount; }
        );
}


void PCAFaceMatcher::calculateEigenFaces(cv::Mat const& differenceMatrix)
{
    auto [eigenValues, eigenVectors] = eigenDecomposition(covarianceMatrix(differenceMatrix));

    eigenVectors = eigenVectors.t();

    if (m_numberOfComponents == 0)
        m_numberOfComponents = eigenVectors.cols;
    else
        m_numberOfComponents = std::min(m_numberOfComponents, eigenVectors.cols);

    m_eigenFaces = multiplyMatrices(
            differenceMatrix,
            eigenVectors.colRange(0, m_numberOfComponents)
        );

    for (int col = 0; col < m_eigenFaces.cols; ++col)
    {
        cv::normalize(
                m_eigenFaces.col(col),
                m_eigenFaces.col(col)
            );
    }
}


void PCAFaceMatcher::calculateProjections(
        std::vector<std::reference_wrapper<FaceImage>> const& trainSet,
        cv::Mat const&                                        differenceMatrix
    )
{
    m_classes.resize(size(trainSet));

    m_projections = cv::Mat::zeros(
            m_eigenFaces.cols,
            static_cast<int>(size(trainSet)),
            CV_64FC1
        );

    cv::Mat const transposedEigenFaces = m_eigenFaces.t();

    for (int col = 0; col < m_projections.cols; ++col)
    {
        m_classes[col] = trainSet[col].get().faceId;

        multiplyMatrices(
                transposedEigenFaces,
                differenceMatrix.col(col)
            ).copyTo(m_projections.col(col));
    }
}


cv::Mat PCAFaceMatcher::getDifferenceMatrix(
        std::vector<std::reference_wrapper<FaceImage>> const& dataSet,
        cv::Mat const&                                        meanImage
    )
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
