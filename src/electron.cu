#include "../include/electron.hpp"

namespace thomson
{
#ifdef USE_CUDA
	template<>
	void electron<float>::normalize2shpere_gpu(cublasHandle_t cublas_handle)
	{
		float norm;
		CUBLAS_CHECK(cublasSnrm2(cublas_handle, dimension_, position_->gpu_data(), 1, &norm));
		norm = 1 / norm;
		CUBLAS_CHECK(cublasSscal(cublas_handle, dimension_, &norm, position_->mutable_gpu_data(), 1));
	}

	template<>
	void electron<double>::normalize2shpere_gpu(cublasHandle_t cublas_handle)
	{
		double norm;
		CUBLAS_CHECK(cublasDnrm2(cublas_handle, dimension_, position_->gpu_data(), 1, &norm));
		norm = 1 / norm;
		CUBLAS_CHECK(cublasDscal(cublas_handle, dimension_, &norm, position_->mutable_gpu_data(), 1));
	}

#endif
}