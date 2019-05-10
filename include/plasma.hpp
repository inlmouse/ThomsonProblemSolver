#pragma once
#ifndef _PLASMA_HPP_
#define _PLASMA_HPP_

#include <glasssix\tensor.hpp>
#include <glasssix\profiler.hpp>

namespace thomson
{
	template<typename Dtype>
	class plasma
	{
		int electorn_num_;
		int dim_;
		int device_;

		glasssix::excalibur::tensor<Dtype>* electorns_ = nullptr;
		glasssix::excalibur::tensor<Dtype>* combine_force_ = nullptr;
		glasssix::excalibur::tensor<Dtype>* component_force_ = nullptr;
		glasssix::excalibur::tensor<Dtype>* distance_ = nullptr;
		glasssix::excalibur::tensor<Dtype>* multiplier_ = nullptr;
		// statical system indicators
		// all potential energy
		Dtype pe_;
		// the centor of all electron positions
		glasssix::excalibur::tensor<Dtype>* center_ = nullptr;
		// the norm of center
		Dtype norm_center_;

#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
#endif

		void normalize2shpere();
#ifdef USE_CUDA
		void normalize2shpere_gpu(Dtype* position_data);

		void calccombineforce_gpu(const Dtype* position_data, Dtype* combine_force_data,
			Dtype* component_force_data, Dtype* distance_data);
#endif

		glasssix::Profiler *profiler;
	public:
		plasma(int e_num, int dim, int device);

		~plasma();

		void Random_Init_Electorns();

		void Init_Electorns_From_File(std::string path);

		void Dump_Electorns_To_File(std::string path);

		void Forward();
		void Backward(Dtype lr);

		Dtype CalculatePotentialEnergy();

		Dtype GetPE() const
		{
			return pe_;
		}

		void SetPE(Dtype pe)
		{
			pe_ = pe;
		}

		//DISABLE_COPY_AND_ASSIGN(plasma);
	};
}

#endif // !_PLASMA_HPP_
