#include <algorithm>

#include <opencv2/core.hpp>

#include "matrixOperations.h"


cv::Mat subtractMatrices(
        cv::Mat const& left,
        cv::Mat const& right
    )
{
    auto resultMatrix = left.clone();

    std::transform(
            resultMatrix.begin<double>(),
            resultMatrix.end<double>(),
            right.begin<double>(),
            resultMatrix.begin<double>(),
            [](auto&& leftValue, auto&& rightValue) { return leftValue - rightValue; }
        );

    return resultMatrix;
}


cv::Mat multiplyMatrices(
        cv::Mat const& left,
        cv::Mat const& right
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


double matricesDistance(
        cv::Mat const& left,
        cv::Mat const& right
    )
{
    auto differenceMatrix = subtractMatrices(
            left,
            right
        );

    cv::pow(
            differenceMatrix,
            2.,
            differenceMatrix
        );

    auto totalDifference = cv::sum(differenceMatrix);

    cv::sqrt(
            totalDifference,
            totalDifference
        );

    return totalDifference[0];
}


std::pair<
        cv::Mat,
        cv::Mat
    > eigenDecomposition(cv::Mat const& matrix)
{
    cv::Mat eigenValues;
    cv::Mat eigenVectors;

    cv::eigen(
            matrix,
            eigenValues,
            eigenVectors
        );

    return {
            std::move(eigenValues),
            std::move(eigenVectors)
        };
}


cv::Mat covarianceMatrix(cv::Mat const& matrix)
{
    return multiplyMatrices(
            matrix.t(),
            matrix
        );
}
