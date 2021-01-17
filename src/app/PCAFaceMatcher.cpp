#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "PCAFaceMatcher.h"
#include "FaceImage.h"
#include "matrixOperations.h"


PCAFaceMatcher::PCAFaceMatcher(int const numberOfComponents) :
    m_numberOfComponents(numberOfComponents)
{ }


void PCAFaceMatcher::train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet)
{
    clear();

    if (empty(trainSet))
        return;

    calculateMeanImage(trainSet);

    auto differenceMatrix = getDifferenceMatrix(
            trainSet,
            m_mean
        );

    calculateEigenFaces(differenceMatrix);
    calculateProjections(
            trainSet,
            differenceMatrix
        );
}


unsigned int PCAFaceMatcher::predict(FaceImage const& entry) const
{
    auto entryProjection = multiplyMatrices(
            m_eigenFaces.t(),
            subtractMatrices(
                    entry.imageData,
                    m_mean
                )
        );

    std::vector<double> distances(m_projections.cols, 0);

    for (int col = 0; col < m_projections.cols; ++col)
    {
        distances[col] = matricesDistance(
                entryProjection,
                m_projections.col(col)
            );
    }

    auto itMinDistance = std::min_element(
            begin(distances),
            end(distances)
        );

    return m_classes[std::distance(
            begin(distances),
            itMinDistance
        )];
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
    auto const& firstImage = dataSet.front().get().imageData;

    m_mean = cv::Mat::zeros(
            firstImage.rows,
            firstImage.cols,
            firstImage.type()
        );

    for (auto const& entryRef : dataSet)
    {
        auto const& entryImage = entryRef.get().imageData;

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

    cv::Mat transposedEigenFaces = m_eigenFaces.t();

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
    auto const& firstImage = dataSet.front().get().imageData;

    cv::Mat differenceMatrix = cv::Mat::zeros(
            firstImage.rows,
            static_cast<int>(size(dataSet)),
            firstImage.type()
        );

    for (int col = 0; col < differenceMatrix.cols; ++col)
    {
        auto const& entryImageData = dataSet[col].get().imageData;

        subtractMatrices(
                entryImageData,
                meanImage
            ).copyTo(differenceMatrix.col(col));
    }

    return differenceMatrix;
}
