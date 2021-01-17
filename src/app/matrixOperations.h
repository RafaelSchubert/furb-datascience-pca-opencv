#if !defined(MATRIXOPERATIONS_H_)
# define MATRIXOPERATIONS_H_


cv::Mat subtractMatrices(
        cv::Mat const& left,
        cv::Mat const& right
    );


cv::Mat multiplyMatrices(
        cv::Mat const& left,
        cv::Mat const& right
    );


double matricesDistance(
        cv::Mat const& left,
        cv::Mat const& right
    );


cv::Mat covarianceMatrix(cv::Mat const& matrix);


std::pair<
        cv::Mat,
        cv::Mat
    > eigenDecomposition(cv::Mat const& matrix);


#endif // #if !defined(MATRIXOPERATIONS_H_)
