#if !defined(PCAFACEMATCHER_H_)
# define PCAFACEMATCHER_H_


struct FaceImage;


class PCAFaceMatcher
{
public:
    PCAFaceMatcher() = default;

    PCAFaceMatcher(int numberOfComponents);

    PCAFaceMatcher(PCAFaceMatcher const&) = delete;
    PCAFaceMatcher(PCAFaceMatcher&&)      = delete;

    virtual ~PCAFaceMatcher() = default;

    PCAFaceMatcher& operator=(PCAFaceMatcher const&) = delete;
    PCAFaceMatcher& operator=(PCAFaceMatcher&&)      = delete;

    void train(std::vector<std::reference_wrapper<FaceImage>> const& trainSet);

    void predict(std::vector<std::reference_wrapper<FaceImage>> const& dataSet) const;

private:
    void clear();

    void calculateMeanImage(std::vector<std::reference_wrapper<FaceImage>> const& trainSet);

    void calculateEigenFaces(cv::Mat const& differenceMatrix);

    void calculateProjections(
            std::vector<std::reference_wrapper<FaceImage>> const& trainSet,
            cv::Mat const&                                        differenceMatrix
        );

    cv::Mat getDifferenceMatrix(
            std::vector<std::reference_wrapper<FaceImage>> const& trainSet,
            cv::Mat const&                                        meanImage
        );

private:
    cv::Mat                   m_mean;
    cv::Mat                   m_eigenFaces;
    cv::Mat                   m_projections;
    std::vector<unsigned int> m_classes;
    int                       m_numberOfComponents = 0;
};


#endif // #if !defined(PCAFACEMATCHER_H_)
