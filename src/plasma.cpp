#include <random>
#include <cblas.h>
#include "../include/plasma.hpp"

using namespace glasssix::excalibur;

namespace thomson
{
	template <typename Dtype>
	plasma<Dtype>::plasma(int e_num, int dim, int device)
	{
		electorn_num_ = e_num;
		dim_ = dim;
		device_ = device;
		CHECK_GE(electorn_num_, dim_);
		electorns_ = new tensor<Dtype>(std::vector<int>{electorn_num_, dim_, 1, 1}, device_);
		combine_force_ = new tensor<Dtype>(std::vector<int>{electorn_num_, dim_, 1, 1}, device_);
		component_force_ = new tensor<Dtype>(std::vector<int>{electorn_num_, dim_, 1, 1}, device_);
		distance_ = new tensor<Dtype>(std::vector<int>{1, electorn_num_, 1, 1}, device_);
		multiplier_ = new tensor<Dtype>(std::vector<int>{1, electorn_num_, 1, 1}, device_);
		memset(distance_->mutable_cpu_data(), 0, electorn_num_ * sizeof(Dtype));
		pe_ = (Dtype)0;
		center_ = new tensor<Dtype>(std::vector<int>{1, dim_, 1, 1}, device_);
		memset(center_->mutable_cpu_data(), 0, dim_ * sizeof(Dtype));
		norm_center_ = (Dtype)0;
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
		//profiler = glasssix:://profiler::Get();
		//profiler->TurnON();
	}

	template <typename Dtype>
	plasma<Dtype>::~plasma()
	{
		Dump_Electorns_To_File("tps_" + std::to_string(electorn_num_) +
			"_" + std::to_string(dim_) + ".dat");
		delete electorns_;
		delete combine_force_;
		delete component_force_;
		delete distance_;
		delete multiplier_;
		delete center_;
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
		//profiler->TurnOFF();
		//profiler->DumpProfile("details.json");
	}

	template<typename Dtype>
	void plasma<Dtype>::Random_Init_Electorns()
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> d(0, 1);
		memset(combine_force_->mutable_cpu_data(), 0, combine_force_->count() * sizeof(Dtype));
		Dtype* position_data = electorns_->mutable_cpu_data();
		for (int i = 0; i < electorns_->count(); i++)
		{
			position_data[i] = static_cast<Dtype>(d(gen));
		}
		normalize2shpere();
		//
		Dtype* multiplier_data = multiplier_->mutable_cpu_data();
		for (int i = 0; i < multiplier_->count(); i++)
		{
			multiplier_data[i] = static_cast<Dtype>(1);
		}
	}

	template<typename Dtype>
	void plasma<Dtype>::Init_Electorns_From_File(std::string path)
	{
		std::ifstream in_stream;
		in_stream.open(path.c_str(), std::ifstream::binary);
		in_stream.read(reinterpret_cast<char*>(electorns_->mutable_cpu_data()), electorns_->count() * sizeof(Dtype));
		in_stream.close();
	}

	template<typename Dtype>
	void plasma<Dtype>::Dump_Electorns_To_File(std::string path)
	{
		std::ofstream out_stream;
		out_stream.open(path.c_str(), std::ofstream::binary);
		out_stream.write(reinterpret_cast<const char*>(electorns_->cpu_data()), electorns_->count() * sizeof(Dtype));
		out_stream.close();
	}

	template<>
	void plasma<float>::normalize2shpere()
	{
		if (device_ >= 0)
		{
			normalize2shpere_gpu(electorns_->mutable_gpu_data());
		}
		else
		{
			float* position_data = electorns_->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < electorn_num_; i++)
			{
				float norm = cblas_snrm2(dim_, position_data + i * dim_, 1);
				cblas_sscal(dim_, 1 / norm, position_data + i * dim_, 1);
			}
		}
	}

	template<>
	void plasma<double>::normalize2shpere()
	{
		if (device_ >= 0)
		{
			normalize2shpere_gpu(electorns_->mutable_gpu_data());
		}
		else
		{
			double* position_data = electorns_->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < electorn_num_; i++)
			{
				double norm = cblas_dnrm2(dim_, position_data + i * dim_, 1);
				cblas_dscal(dim_, 1 / norm, position_data + i * dim_, 1);
			}
		}
	}

	template<>
	void plasma<float>::Backward(float lr)
	{
		if (device_ >= 0)
		{
			float* position_data = electorns_->mutable_gpu_data();
			float* combine_force_data = combine_force_->mutable_gpu_data();
			CUBLAS_CHECK(cublasSaxpy(cublas_handle_, dim_ * electorn_num_, &lr,
				combine_force_data, 1, position_data, 1));
			normalize2shpere();
			CUDA_CHECK(cudaMemset(combine_force_data, 0, dim_ * electorn_num_ * sizeof(float)));
		}
		else
		{
			float* position_data = electorns_->mutable_cpu_data();
			float* combine_force_data = combine_force_->mutable_cpu_data();
			cblas_saxpy(dim_ * electorn_num_, 1.0*lr, combine_force_data, 1, position_data, 1);
			normalize2shpere();
			memset(combine_force_data, 0, dim_ * electorn_num_ * sizeof(float));
		}
	}

	template<>
	void plasma<double>::Backward(double lr)
	{
		if (device_ >= 0)
		{
			double* position_data = electorns_->mutable_gpu_data();
			double* combine_force_data = combine_force_->mutable_gpu_data();
			CUBLAS_CHECK(cublasDaxpy(cublas_handle_, dim_ * electorn_num_, &lr,
				combine_force_data, 1, position_data, 1));
			normalize2shpere();
			CUDA_CHECK(cudaMemset(combine_force_data, 0, dim_ * electorn_num_ * sizeof(double)));
		}
		else
		{
			double* position_data = electorns_->mutable_cpu_data();
			double* combine_force_data = combine_force_->mutable_cpu_data();
			cblas_daxpy(dim_ * electorn_num_, 1.0*lr, combine_force_data, 1, position_data, 1);
			normalize2shpere();
			memset(combine_force_data, 0, dim_ * electorn_num_);
		}
	}

	template<>
	void plasma<float>::Forward()
	{
		if (device_ >= 0)
		{
			const float* position_data = electorns_->cpu_data();
			float* combine_force_data = combine_force_->mutable_cpu_data();
			float* component_force_data = component_force_->mutable_cpu_data();
			float* distance_data = distance_->mutable_cpu_data();
			CUDA_CHECK(cudaMemset(distance_data, 0, distance_->count() * sizeof(float)));
			calccombineforce_gpu(position_data, combine_force_data, component_force_data, distance_data);
		}
		else
		{
			const float* position_data = electorns_->cpu_data();
			float* combine_force_data = combine_force_->mutable_cpu_data();
			float* component_force_data = component_force_->mutable_cpu_data();
			float* distance_data = distance_->mutable_cpu_data();
			const float* multiplier_data = multiplier_->cpu_data();
			memset(distance_data, 0, distance_->count() * sizeof(float));
			for (int i = 0; i < electorn_num_; i++)
			{
				//profiler->ScopeStart("copy");
				memcpy(component_force_data, position_data, electorn_num_ * dim_ * sizeof(float));
				//profiler->ScopeEnd();
				float* combine_force_data_i_ptr = combine_force_data + dim_ * i;
				const float* position_data_i_ptr = position_data + dim_ * i;
				//profiler->ScopeStart("kernel");
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int j = 0; j < electorn_num_; j++)
				{
					float* component_force_data_j_ptr = component_force_data + dim_ * j;
					if (i != j)
					{
						cblas_saxpby(dim_, 1.0, position_data_i_ptr, 1, -1.0, component_force_data_j_ptr, 1);
						float f_norm = cblas_snrm2(dim_, component_force_data_j_ptr, 1);
						distance_data[j] += 1.0 / f_norm;
						cblas_sscal(dim_, 1 / (f_norm*f_norm), component_force_data_j_ptr, 1);
					}
					else
					{
						memset(component_force_data_j_ptr, 0, dim_ * sizeof(float));
					}
				}
				//profiler->ScopeEnd();
				//profiler->ScopeStart("gemv");
				cblas_sgemv(CblasRowMajor, CblasTrans, electorn_num_, dim_, 1.0, component_force_data,
					dim_, multiplier_data, 1, 0.0, combine_force_data_i_ptr, 1);
				//profiler->ScopeEnd();
			}
		}
	}

	template<>
	void plasma<double>::Forward()
	{
		if (device_ >= 0)
		{
			const double* position_data = electorns_->gpu_data();
			double* combine_force_data = combine_force_->mutable_gpu_data();
			double* component_force_data = component_force_->mutable_gpu_data();
			double* distance_data = distance_->mutable_gpu_data();
			CUDA_CHECK(cudaMemset(distance_data, 0, distance_->count() * sizeof(double)));
			calccombineforce_gpu(position_data, combine_force_data, component_force_data, distance_data);
		}
		else
		{
			const double* position_data = electorns_->cpu_data();
			double* combine_force_data = combine_force_->mutable_cpu_data();
			double* component_force_data = component_force_->mutable_cpu_data();
			double* distance_data = distance_->mutable_cpu_data();
			const double* multiplier_data = multiplier_->cpu_data();
			memset(distance_data, 0, distance_->count() * sizeof(double));
			for (int i = 0; i < electorn_num_; i++)
			{
				//profiler->ScopeStart("copy");
				memcpy(component_force_data, position_data, electorn_num_ * dim_ * sizeof(double));
				//profiler->ScopeEnd();
				double* combine_force_data_i_ptr = combine_force_data + dim_ * i;
				const double* position_data_i_ptr = position_data + dim_ * i;
				//profiler->ScopeStart("kernel");
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int j = 0; j < electorn_num_; j++)
				{
					double* component_force_data_j_ptr = component_force_data + dim_ * j;
					if (i != j)
					{
						cblas_daxpby(dim_, 1.0, position_data_i_ptr, 1, -1.0, component_force_data_j_ptr, 1);
						double f_norm = cblas_dnrm2(dim_, component_force_data_j_ptr, 1);
						distance_data[j] += 1.0 / f_norm;
						cblas_dscal(dim_, 1.0 / (f_norm*f_norm), component_force_data_j_ptr, 1);
					}
					else
					{
						memset(component_force_data_j_ptr, 0, dim_ * sizeof(double));
					}
				}
				//profiler->ScopeEnd();
				//profiler->ScopeStart("gemv");
				cblas_dgemv(CblasRowMajor, CblasTrans, electorn_num_, dim_, 1.0, component_force_data,
					dim_, multiplier_data, 1, 0.0, combine_force_data_i_ptr, 1);
				//profiler->ScopeEnd();
			}
		}
	}

	template<>
	float plasma<float>::CalculatePotentialEnergy()
	{
		if (device_ >=0 )
		{
			float res = 0.0f;
			CUBLAS_CHECK(cublasSdot(cublas_handle_, electorn_num_, distance_->gpu_data(), 1, multiplier_->gpu_data(), 1, &res));
			return res / 2;
		}
		else
		{
			return cblas_sdot(electorn_num_, distance_->cpu_data(), 1, multiplier_->cpu_data(), 1) / 2;
		}
	}

	template<>
	double plasma<double>::CalculatePotentialEnergy()
	{
		if (device_ >= 0)
		{
			double res = 0.0;
			CUBLAS_CHECK(cublasDdot(cublas_handle_, electorn_num_, distance_->gpu_data(), 1, multiplier_->gpu_data(), 1, &res));
			return res / 2;
		}
		else
		{
			return cblas_ddot(electorn_num_, distance_->cpu_data(), 1, multiplier_->cpu_data(), 1) / 2;
		}
	}



	template class plasma<float>;
	template class plasma<double>;
}