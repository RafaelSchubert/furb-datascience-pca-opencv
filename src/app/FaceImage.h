#if !defined(FACEIMAGE_H_)
# define FACEIMAGE_H_


namespace cv
{
    class Mat;
}


struct FaceImage
{
    cv::Mat      imageData;
    unsigned int imageId = 0;
    unsigned int faceId  = 0;
};


#endif // #if !defined(FACEIMAGE_H_)
