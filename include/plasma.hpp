#pragma once
#ifndef _PLASMA_HPP_
#define _PLASMA_HPP_

#include "electron.hpp"
#include <vector>

namespace thomson
{
	template<typename Dtype>
	class plasma
	{
		std::vector<electron<Dtype>*> electorns_;
		int electorn_num_;
		int dim_;
		int device_;

		// statical system indicators
		// all potential energy
		Dtype pe_;
		// the centor of all electron positions
		tensor<Dtype>* center_;
		// the norm of center
		Dtype norm_center_;

#ifdef _OPENMP
		// the full 1 array for blas multi with (electorn_num - 1) length
		tensor<Dtype>* multiplier_;
		//Pairwise PotentialEnergy, a (electorn_num - 1) length array
		tensor<Dtype>* pairwise_pe_;
		//The sum of Pairwise PotentialEnergy for each cols, a (electorn_num - 1) length array
		tensor<Dtype>* pairwise_pe_sum_;
#endif

		//profiler
		//Timer profiler;
	public:
		plasma(int e_num, int dim, int device);
		~plasma();

		void CalculateAllForce();

		void UpdateAllPosition(Dtype lr = (Dtype)1);

		Dtype CalculatePotentialEnergy();

#ifdef _OPENMP
		Dtype CalculatePotentialEnergy_Parallel();
#endif

		void StartingOptimization(bool fast_pe_calculation = false);

	private:
		void calculate_all_force_cpu();
		void update_all_position_cpu(Dtype lr);
		Dtype calculate_potential_energy_cpu();

#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;

		void calculate_all_force_gpu();
		void update_all_position_gpu(Dtype lr);
		Dtype calculate_potential_energy_gpu();
#endif
	};
}

#endif