#include "../include/plasma.hpp"

namespace thomson
{
#ifdef USE_CUDA
	template <typename Dtype>
	void plasma<Dtype>::calculate_all_force_gpu()
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
					electorns_[i]->add1componentforce_gpu(cublas_handle_, electorns_[j]->getcurrentposition()->gpu_data());
				}
			}
		}
	}

	template <typename Dtype>
	void plasma<Dtype>::update_all_position_gpu(Dtype lr)
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < electorn_num_; i++)
		{
			electorns_[i]->updateposition_gpu(cublas_handle_, lr);
		}
	}

	template <typename Dtype>
	Dtype plasma<Dtype>::calculate_potential_energy_gpu()
	{
		//E = \sum_{i<j}\frac{1}{r_{ij}}
		Dtype PE = 0;
		for (int i = 0; i < electorn_num_; i++)
		{
			for (int j = i + 1; j < electorn_num_; j++)
			{
				PE += 1 / electorns_[i]->calculatedistance_gpu(cublas_handle_, electorns_[j]->getcurrentposition()->gpu_data());
			}
		}
		return PE;
	}

	template class plasma<float>;
	template class plasma<double>;
#endif
}