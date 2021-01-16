#if !defined(PCAFACEMATCHER_H_)
# define PCAFACEMATCHER_H_


struct FaceImage;


class PCAFaceMatcher
{
public:
    void train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet);

    void predict(std::vector<std::reference_wrapper<FaceImage>> const& dataSet) const;

private:
    static cv::Mat multiplyMatrices(
            cv::Mat left,
            cv::Mat right
        );

    cv::Mat getMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& trainSet) const;

    cv::Mat getDifferenceMatrix(
            std::vector<std::reference_wrapper<FaceImage>> const& trainSet,
            cv::Mat const&                                        meanImage
        ) const;

    cv::Mat getCovarianceMatrix(cv::Mat const& matrix) const;

private:
    cv::Mat m_projections;
};


#endif // #if !defined(PCAFACEMATCHER_H_)
