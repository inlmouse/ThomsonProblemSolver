#pragma once
#ifndef	_ELECTORN_HPP_
#define _ELECTORN_HPP_
#include "tensor.hpp"

namespace thomson
{
	template<typename Dtype>
	class electron
	{
		tensor<Dtype>* position_;
		int dimension_;
		tensor<Dtype>* combine_force_;
		int device_;

	public:
		electron(int dimension, int device);
		~electron();
		electron(const electron& e);

		const electron& operator=(const electron& e);
		void normalize2shpere_cpu();
		void combineforce2zero_cpu();
		void add1componentforce_cpu(const Dtype* other_position);
		Dtype calculatedistance_cpu(const Dtype* other_position);
		void updateposition_cpu(Dtype lr);
#ifdef USE_CUDA
		void normalize2shpere_gpu(cublasHandle_t cublas_handle);
		void combineforce2zero_gpu();
		void add1componentforce_gpu(cublasHandle_t cublas_handle, const Dtype* other_position);
		Dtype calculatedistance_gpu(cublasHandle_t cublas_handle, const Dtype* other_position);
		void updateposition_gpu(cublasHandle_t cublas_handle, Dtype lr);
#endif

		const tensor<Dtype>* getcurrentposition();
	};
}

#endif // _ELECTORN_HPP_