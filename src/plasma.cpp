#include "../include/plasma.hpp"
#include <iostream>
#include <cblas.h>

namespace thomson
{
	template <typename Dtype>
	plasma<Dtype>::plasma(int e_num, int dim, int device)
	{
		electorn_num_ = e_num;
		device_ = device;
		this->dim_ = dim;
		for (int i = 0; i < e_num; i++)
		{
			electorns_.push_back(new electron<Dtype>(dim, device_));
		}
		pe_ = (Dtype)0;
		center_ = new tensor<Dtype>(std::vector<int>{dim_}, device_);
		memset(center_->mutable_cpu_data(), 0, dim_ * sizeof(Dtype));
		norm_center_ = (Dtype)0;
#ifdef _OPENMP
		//optional, for parallel calculation
		multiplier_ = new tensor<Dtype>(std::vector<int>{electorn_num_ - 1}, device_);
		Dtype* multiplier_data = multiplier_->mutable_cpu_data();
		for (int i = 0; i < electorn_num_ - 1; i++)
		{
			multiplier_data[i] = (Dtype)1;
		}
		pairwise_pe_ = new tensor<Dtype>(std::vector<int>{electorn_num_ - 1}, device_);
		memset(pairwise_pe_->mutable_cpu_data(), 0, (electorn_num_ - 1) * sizeof(Dtype));
		pairwise_pe_sum_ = new tensor<Dtype>(std::vector<int>{electorn_num_ - 1}, device_);
		memset(pairwise_pe_sum_->mutable_cpu_data(), 0, (electorn_num_ - 1) * sizeof(Dtype));
#endif
		// create cublas handle
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#else
			NO_GPU;
#endif
		}
	}

	template <typename Dtype>
	plasma<Dtype>::~plasma()
	{
		electorns_.clear();
		delete center_;
#ifdef _OPENMP
		//optional, for parallel calculation
		delete multiplier_;
		delete pairwise_pe_;
		delete pairwise_pe_sum_;
#endif
		//delete cublas handle
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#else
			NO_GPU;
#endif
		}
	}

	template <typename Dtype>
	void plasma<Dtype>::CalculateAllForce()
	{
		if (device_>=0)
		{
			calculate_all_force_gpu();
		}
		else
		{
			calculate_all_force_cpu();
		}
	}

	template <typename Dtype>
	void plasma<Dtype>::UpdateAllPosition(Dtype lr)
	{
		if (device_ >= 0)
		{
			update_all_position_gpu(lr);
		}
		else
		{
			update_all_position_cpu(lr);
		}
	}

	template <typename Dtype>
	Dtype plasma<Dtype>::CalculatePotentialEnergy()
	{
		if (device_ >= 0)
		{
			return calculate_potential_energy_gpu();
		}
		else
		{
			return calculate_potential_energy_cpu();
		}
	}

#ifdef _OPENMP
	template <>
	float plasma<float>::CalculatePotentialEnergy_Parallel()
	{
		float* pairwise_pe_sum_data = pairwise_pe_sum_->mutable_cpu_data();
		for (int i = 0; i < electorn_num_ - 1; i++)
		{
			float* pairwise_pe_data = pairwise_pe_->mutable_cpu_data();
#pragma omp parallel for
			for (int j = i + 1; j < electorn_num_; j++)
			{
				pairwise_pe_data[j - (i + 1)] = 1 / electorns_[i]->calculatedistance_cpu(electorns_[j]->getcurrentposition()->cpu_data());
			}
			pairwise_pe_sum_data[i] = cblas_sdot(electorn_num_ - 1 - i, pairwise_pe_->cpu_data(), 1, multiplier_->cpu_data(), 1);
		}
		return cblas_sdot(electorn_num_ - 1, pairwise_pe_sum_->cpu_data(), 1, multiplier_->cpu_data(), 1);
	}

	template <>
	double plasma<double>::CalculatePotentialEnergy_Parallel()
	{
		double* pairwise_pe_sum_data = pairwise_pe_sum_->mutable_cpu_data();
		for (int i = 0; i < electorn_num_ - 1; i++)
		{
			double* pairwise_pe_data = pairwise_pe_->mutable_cpu_data();
#pragma omp parallel for
			for (int j = i + 1; j < electorn_num_; j++)
			{
				pairwise_pe_data[j - (i + 1)] = 1 / electorns_[i]->calculatedistance_cpu(electorns_[j]->getcurrentposition()->cpu_data());
			}
			pairwise_pe_sum_data[i] = cblas_ddot(electorn_num_ - 1 - i, pairwise_pe_->cpu_data(), 1, multiplier_->cpu_data(), 1);
		}
		return cblas_ddot(electorn_num_ - 1, pairwise_pe_sum_->cpu_data(), 1, multiplier_->cpu_data(), 1);
	}
#endif

	//private funstions:
	template <typename Dtype>
	void plasma<Dtype>::calculate_all_force_cpu()
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < electorn_num_; i++)
		{
			for (int j = 0; j < electorn_num_; j++)
			{
				if (i != j)
				{
					electorns_[i]->add1componentforce_cpu(electorns_[j]->getcurrentposition()->cpu_data());
				}
			}
		}
	}

	template <typename Dtype>
	void plasma<Dtype>::update_all_position_cpu(Dtype lr)
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < electorn_num_; i++)
		{
			electorns_[i]->updateposition_cpu(lr);
		}
	}

	template <typename Dtype>
	Dtype plasma<Dtype>::calculate_potential_energy_cpu()
	{
		//E = \sum_{i<j}\frac{1}{r_{ij}}
		Dtype PE = 0;
		for (int i = 0; i < electorn_num_; i++)
		{
			for (int j = i + 1; j < electorn_num_; j++)
			{
				PE += 1 / electorns_[i]->calculatedistance_cpu(electorns_[j]->getcurrentposition()->cpu_data());
			}
		}
		return PE;
	}

	template class plasma<float>;
	template class plasma<double>;
}
