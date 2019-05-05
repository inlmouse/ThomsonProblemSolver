#include "../include/electron.hpp"
#include <iostream>

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

	template <typename Dtype>
	void electron<Dtype>::combineforce2zero_gpu()
	{
		CUDA_CHECK(cudaMemset(combine_force_->mutable_gpu_data(), 0, sizeof(Dtype) * dimension_));
	}

	template<>
	void electron<float>::add1componentforce_gpu(cublasHandle_t cublas_handle, const float* other_position)
	{
		tensor<float>* f_i = new tensor<float>(std::vector<int>{dimension_}, device_);
		CUDA_CHECK(cudaMemcpy(f_i->mutable_gpu_data(), other_position, dimension_ * sizeof(float), cudaMemcpyDeviceToDevice));
		float inverser = -1.0;
		CUBLAS_CHECK(cublasSscal(cublas_handle, dimension_, &inverser, f_i->mutable_gpu_data(), 1));
		float alpha = 1.0;
		CUBLAS_CHECK(cublasSaxpy(cublas_handle, dimension_, &alpha, position_->gpu_data(), 1, f_i->mutable_gpu_data(), 1));
		float f_norm;
		CUBLAS_CHECK(cublasSnrm2(cublas_handle, dimension_, f_i->gpu_data(), 1, &f_norm));
		f_norm = 1 / (f_norm * f_norm);
		CUBLAS_CHECK(cublasSscal(cublas_handle, dimension_, &f_norm, f_i->mutable_gpu_data(), 1));
		CUBLAS_CHECK(cublasSaxpy(cublas_handle, dimension_, &alpha, f_i->gpu_data(), 1, combine_force_->mutable_gpu_data(), 1));
		delete f_i;
	}

	template<>
	void electron<double>::add1componentforce_gpu(cublasHandle_t cublas_handle, const double* other_position)
	{
		double* temp = new double[3];
		tensor<double>* f_i = new tensor<double>(std::vector<int>{dimension_}, device_); 
		CUDA_CHECK(cudaMemcpy(f_i->mutable_gpu_data(), other_position, dimension_ * sizeof(double), cudaMemcpyDeviceToDevice));
		double inverser = -1.0;
		cublasGetVector(dimension_, sizeof(double), f_i->gpu_data(), 1, temp, 1);
		CUBLAS_CHECK(cublasDscal(cublas_handle, dimension_, &inverser, f_i->mutable_gpu_data(), 1));
		double alpha = 1.0;
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, dimension_, &alpha, position_->gpu_data(), 1, f_i->mutable_gpu_data(), 1));
		double f_norm;
		CUBLAS_CHECK(cublasDnrm2(cublas_handle, dimension_, f_i->gpu_data(), 1, &f_norm));
		f_norm = 1 / (f_norm * f_norm);
		CUBLAS_CHECK(cublasDscal(cublas_handle, dimension_, &f_norm, f_i->mutable_gpu_data(), 1));
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, dimension_, &alpha, f_i->gpu_data(), 1, combine_force_->mutable_gpu_data(), 1));
		delete f_i;
	}

	template<>
	float electron<float>::calculatedistance_gpu(cublasHandle_t cublas_handle, const float* other_position)
	{
		tensor<float>* f_i = new tensor<float>(std::vector<int>{dimension_}, device_);
		CUDA_CHECK(cudaMemcpy(f_i->mutable_gpu_data(), other_position, dimension_ * sizeof(float), cudaMemcpyDefault));
		float inverser = -1.0;
		CUBLAS_CHECK(cublasSscal(cublas_handle, dimension_, &inverser, f_i->mutable_gpu_data(), 1));
		float alpha = 1.0;
		CUBLAS_CHECK(cublasSaxpy(cublas_handle, dimension_, &alpha, position_->gpu_data(), 1, f_i->mutable_gpu_data(), 1));
		float f_norm;
		CUBLAS_CHECK(cublasSnrm2(cublas_handle, dimension_, f_i->gpu_data(), 1, &f_norm));
		delete f_i;
		return f_norm;
	}

	template<>
	double electron<double>::calculatedistance_gpu(cublasHandle_t cublas_handle, const double* other_position)
	{
		tensor<double>* f_i = new tensor<double>(std::vector<int>{dimension_}, device_);
		CUDA_CHECK(cudaMemcpy(f_i->mutable_gpu_data(), other_position, dimension_ * sizeof(double), cudaMemcpyDefault));
		double inverser = -1.0;
		CUBLAS_CHECK(cublasDscal(cublas_handle, dimension_, &inverser, f_i->mutable_gpu_data(), 1));
		double alpha = 1.0;
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, dimension_, &alpha, position_->gpu_data(), 1, f_i->mutable_gpu_data(), 1));
		double f_norm;
		CUBLAS_CHECK(cublasDnrm2(cublas_handle, dimension_, f_i->gpu_data(), 1, &f_norm));
		delete f_i;
		return f_norm;
	}

	template <>
	void electron<float>::updateposition_gpu(cublasHandle_t cublas_handle, float lr)
	{
		CUBLAS_CHECK(cublasSaxpy(cublas_handle, dimension_, &lr, combine_force_->gpu_data(), 1, position_->mutable_gpu_data(), 1));
		normalize2shpere_gpu(cublas_handle);
		CUDA_CHECK(cudaMemset(combine_force_->mutable_gpu_data(), 0, sizeof(float) * dimension_));
	}

	template <>
	void electron<double>::updateposition_gpu(cublasHandle_t cublas_handle, double lr)
	{
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, dimension_, &lr, combine_force_->gpu_data(), 1, position_->mutable_gpu_data(), 1));
		normalize2shpere_gpu(cublas_handle);
		CUDA_CHECK(cudaMemset(combine_force_->mutable_gpu_data(), 0, sizeof(double) * dimension_));
	}

	template class electron<float>;
	template class electron<double>;
#endif
}