#include "DeepClassifierTorch.hpp"

#include "openMVG/features/tbmr/tbmr.hpp"
#include "openMVG/image/image_converter.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <opencv2/core/eigen.hpp>

#include <cunn.h>
#include <cmath>
#include <iostream>
#include <THC/THC.h>

using namespace openMVG::features;
using namespace openMVG::image;

template <typename Image>
void NormalizePatch
(
  const Image & src_img ,
  const AffinePointFeature & feat ,
  const int patch_size ,
  Image & out_patch
)
{
  // Mapping function
  Eigen::Matrix<double,2,2> A;
  A << feat.a(), feat.b(),
       feat.b(), feat.c();

  // Inverse square root
  A = A.pow( -0.5 ) ;

  const float sc = 2.f * 3.f / static_cast<float>(patch_size);
  A = A * sc ;

  const float half_width = static_cast<float>( patch_size ) / 2.f ;

  // Compute sampling grid
  std::vector< std::pair<float,float> > sampling_grid ;
  sampling_grid.reserve(patch_size*patch_size);
  for( int i = 0 ; i < patch_size ; ++i )
  {
    for( int j = 0 ; j < patch_size ; ++j )
    {
      // Apply transformation relative to the center of the patch (assume origin at 0,0 then map to (x,y) )
      Vec2 pos;
      pos << static_cast<float>( j ) - half_width, static_cast<float>( i ) - half_width ;
      // Map (ie: ellipse transform)
      const Vec2 affineAdapted = A * pos;

      sampling_grid.emplace_back( affineAdapted(1) + feat.y() , affineAdapted(0) + feat.x() );
    }
  }

  Sampler2d< SamplerLinear > sampler ;

  // Sample input image to generate patch
  GenericRessample(
    src_img, sampling_grid,
    patch_size, patch_size,
    sampler,
    out_patch);
}

// So we can cache results
const float sizeOfEllipse(const float l1, const float l2) __attribute__((pure));
const float sizeOfEllipse(const float l1, const float l2) {

	//return 2 * std::sqrt(l1 * l2);
	return 64.0f;
}

#define M 64
void extractPatches(const cv::Mat& image,
    const std::vector<features::AffinePointFeature>& kp,
    std::vector<cv::Mat>& patches) 
{
	for (auto &it : kp) {
    cv::Mat patch(M, M, CV_32F);
    cv::Mat buf;
		const float size = sizeOfEllipse(it.l1(), it.l2());
		//std::cout << "Size of ellipse: " << size << " at point: " << it.x() << ", " << it.y() << " with size: " << it.a() << ", " << it.b() << ", " << it.c() << ", " << it.l1() << ", " << it.l2() <<  std::endl;
    // increase the size of the region to include some context
    cv::getRectSubPix(image, cv::Size(size*1.3, size*1.3), cv::Point2f(it.x(), it.y()), buf);
    cv::Scalar m = cv::mean(buf);
		//std::cout << "Mean: " << m[0] << "\nWidth: " << buf.cols << "\nHeight: " << buf.rows << std::endl;
    cv::resize(buf, patch, cv::Size(M,M));
    patch.convertTo(patch, CV_32F, 1./255.);
    patch = patch.isContinuous() ? patch : patch.clone();
    // mean subtraction is crucial!
    patches.push_back(patch - m[0]/255.);
	}
}

void DeepClassifierTorch::extractDescriptors(std::vector<cv::Mat>& patches)
{
  const size_t batch_size = 128;
  const size_t N = patches.size();

  THFloatTensor *buffer = THFloatTensor_newWithSize4d(batch_size, 1, M, M);
  THCudaTensor *input = THCudaTensor_newWithSize4d(m_state, batch_size, 1, M, M);

  for(int j=0; j < ceil((float)N/batch_size); ++j)
  {
    float *data = THFloatTensor_data(buffer);
    size_t k = 0;
    for(size_t i = j*batch_size; i < std::min((j+1)*batch_size, N); ++i, ++k)
      memcpy(data + k*M*M, patches[i].data, sizeof(float) * M * M);

    THCudaTensor_copyFloat(m_state, input, buffer);
    
	// propagate through the network
    THCudaTensor *output = m_net->forward(input);

	THFloatTensor *desc = THFloatTensor_newWithSize2d(output->size[0], output->size[1]);
	THFloatTensor_copyCuda(m_state, desc, output);

    const size_t feature_dim = output->size[1];

	if (m_descriptors.cols != feature_dim || m_descriptors.rows != N) {
		std::cout << "Feature dims: " << feature_dim << std::endl << "Size of patches: " << N << std::endl;
		m_descriptors.create(N, feature_dim, CV_32F);
	}

    memcpy(m_descriptors.data + j * feature_dim * batch_size * sizeof(float),
        THFloatTensor_data(desc),
        sizeof(float) * feature_dim * k);
    THFloatTensor_free(desc);
  }

  THCudaTensor_free(m_state, input);
  THFloatTensor_free(buffer);
}

void DeepClassifierTorch::extractDescriptorsOpenMVG(std::vector<Image<float>>& patches)
{
  const size_t batch_size = 128;
  const size_t N = patches.size();

  THFloatTensor *buffer = THFloatTensor_newWithSize4d(batch_size, 1, M, M);
  THCudaTensor *input = THCudaTensor_newWithSize4d(m_state, batch_size, 1, M, M);

  for(int j=0; j < ceil((float)N/batch_size); ++j)
  {
    float *data = THFloatTensor_data(buffer);
    size_t k = 0;
    for(size_t i = j*batch_size; i < std::min((j+1)*batch_size, N); ++i, ++k)
      memcpy(data + k*M*M, patches[i].data(), sizeof(float) * M * M);

    THCudaTensor_copyFloat(m_state, input, buffer);
    
	// propagate through the network
    THCudaTensor *output = m_net->forward(input);

	THFloatTensor *desc = THFloatTensor_newWithSize2d(output->size[0], output->size[1]);
	THFloatTensor_copyCuda(m_state, desc, output);

    const size_t feature_dim = output->size[1];

	if (m_descriptorsOpenMVG.cols() != feature_dim || m_descriptorsOpenMVG.rows() != N) {
		std::cout << "Feature dims: " << feature_dim << std::endl << "Size of patches: " << N << std::endl;
		m_descriptorsOpenMVG.resize(N, feature_dim);
	}

    memcpy(m_descriptorsOpenMVG.data() + j * feature_dim * batch_size * sizeof(float),
        THFloatTensor_data(desc),
        sizeof(float) * feature_dim * k);

    THFloatTensor_free(desc);
  }

  THCudaTensor_free(m_state, input);
  THFloatTensor_free(buffer);
}

DeepClassifierTorch::DeepClassifierTorch(const std::string& filename) {
	std::cout << "From inside DeepClassifier, opening: " << filename << std::endl;
	m_state = (THCState*)malloc(sizeof(THCState));
	THCudaInit(m_state);
	m_net = loadNetwork(m_state, filename.c_str());

	m_descriptors = cv::Mat::zeros(1, 1, CV_32F);

}
// TODO: BROKEN
const std::vector<features::AffinePointFeature>& DeepClassifierTorch::extractKeypointsOpenMVG(const Image<unsigned char>& image) {
	std::vector<features::AffinePointFeature> feats_tbmr_bright;
	std::vector<features::AffinePointFeature> feats_tbmr_dark;

	//tbmr::Extract_tbmr(image, feats_tbmr_bright, std::less<uint8_t>());
	//tbmr::Extract_tbmr(image, feats_tbmr_dark, std::greater<uint8_t>());
	//feats_tbmr_bright.insert(feats_tbmr_bright.end(), feats_tbmr_dark.begin(), feats_tbmr_dark.end());
	
	const std::vector<features::AffinePointFeature>& feats_tbmr(feats_tbmr_bright);
	
	return feats_tbmr;
}

float* DeepClassifierTorch::describeOpenMVG(const openMVG::image::Image<unsigned char>& image, const std::vector<openMVG::features::AffinePointFeature>& keypoints) {
	std::vector<cv::Mat> patches;

	cv::Mat imageMat;
	cv::eigen2cv(image, imageMat);
	std::cout << "Converted eigen to cv" << std::endl;
	/*for (auto it : keypoints) {
		Image<unsigned char> imagePatch;
		NormalizePatch(image, it, M, imagePatch);

		cv::Mat imagePatchMat;
		cv::eigen2cv(imagePatch, imagePatchMat);
		cv::Scalar m = cv::mean(imagePatchMat);
		cv::Mat imagePatchMatFloat(M, M, CV_32F);

		imagePatchMatFloat.convertTo(imagePatchMatFloat, CV_32F, 1./255.);
    imagePatchMatFloat = imagePatchMatFloat.isContinuous() ? imagePatchMatFloat : imagePatchMatFloat.clone();
    // mean subtraction is crucial!*/
   /* patches.push_back(imagePatchMat);
	}*/

  extractPatches(imageMat, keypoints, patches);
	
	std::cout << "Extracted all patches" << std::endl;
	extractDescriptors(patches);

	std::cout << "Extracted all descriptors" << std::endl;

	return (float*)(m_descriptors.data);
}
	
const std::vector<DeepClassifierKeypoint>& DeepClassifierTorch::extractKeypoints(const Image<unsigned char>& image) {
	auto feats = extractKeypointsOpenMVG(image);

	std::vector<DeepClassifierKeypoint> deepClassifierKeypoints;
	for (auto it : feats) {
		deepClassifierKeypoints.push_back(
			DeepClassifierKeypoint(
				it.x(), it.y(),
				it.l1(), it.l2(),
				it.orientation(),
				it.a(), it.b(), it.c()
			)
		);
	}
	const std::vector<DeepClassifierKeypoint>& constDeepClassifierKeypoints(deepClassifierKeypoints);
	return constDeepClassifierKeypoints;
}

float* DeepClassifierTorch::describe(const openMVG::image::Image<unsigned char>& image, const std::vector<DeepClassifierKeypoint>& keypoints) {
	Image<unsigned char> Icpy (image);
	std::vector<Image<float>> patches;

	for (auto it : keypoints) {
		Image<unsigned char> patch;
		const openMVG::features::AffinePointFeature affinePointFeature(it.x, it.y, it.a, it.b, it.c);
		NormalizePatch(Icpy, affinePointFeature, 64, patch);
		const auto mean = patch.mean();
		Image<float> patchFloat;
		image::unsignedChar2Float(patch, &patchFloat, mean / 255.f);
		patches.push_back(patchFloat);
	}
	
	//extractDescriptors(patches);

	return m_descriptorsOpenMVG.data();
}

DeepClassifierTorch::~DeepClassifierTorch() {
	THCudaShutdown(m_state);
}
