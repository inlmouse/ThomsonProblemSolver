#include "../include/plasma.hpp"

namespace thomson
{
#ifdef USE_CUDA
	__global__ void normalize2shpere_kernel(float* position_data, int electorn_num_, int dim_)
	{
		CUDA_KERNEL_LOOP(index, electorn_num_)
		{
			float norm = 0.0f;
			for (int j = 0; j < dim_; j++)
			{
				float temp = position_data[index * dim_ + j];
				norm = fmaf(temp, temp, norm);
			}
			norm = 1.0f / sqrtf(norm);
			for (int j = 0; j < dim_; j++)
			{
				position_data[index * dim_ + j] = position_data[index * dim_ + j] * norm;
			}
		}
	}

	__global__ void normalize2shpere_kernel(double* position_data, int electorn_num_, int dim_)
	{
		CUDA_KERNEL_LOOP(index, electorn_num_)
		{
			double norm = 0.0f;
			for (int j = 0; j < dim_; j++)
			{
				double temp = position_data[index * dim_ + j];
				norm += temp * temp;
			}
			norm = 1.0f / sqrt(norm);
			for (int j = 0; j < dim_; j++)
			{
				position_data[index * dim_ + j] = position_data[index * dim_ + j] * norm;
			}
		}
	}

	template<>
	void plasma<float>::normalize2shpere_gpu(float* position_data)
	{
		cudaSetDevice(device_);
		normalize2shpere_kernel << <glasssix::excalibur::CUDA_GET_BLOCKS(electorn_num_), glasssix::excalibur::CUDA_NUM_THREADS >> >
			(position_data, electorn_num_, dim_);
		CUDA_POST_KERNEL_CHECK;
	}

	template<>
	void plasma<double>::normalize2shpere_gpu(double* position_data)
	{
		cudaSetDevice(device_);
		normalize2shpere_kernel << <glasssix::excalibur::CUDA_GET_BLOCKS(electorn_num_), glasssix::excalibur::CUDA_NUM_THREADS >> >
			(position_data, electorn_num_, dim_);
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ void calccombineforce_kernel(int electorn_num_, int dim_, int x,
		const float* position_data_i_ptr, float* component_force_data, float* distance_data)
	{
		CUDA_KERNEL_LOOP(index, electorn_num_)
		{
			float* component_force_data_j_ptr = component_force_data + dim_ * index;
			if (x != index)
			{
				float f_norm = 0.0f;
				for (int d = 0; d < dim_; d++)
				{
					component_force_data_j_ptr[d] = component_force_data_j_ptr[d] - position_data_i_ptr[d];
					f_norm += component_force_data_j_ptr[d] * component_force_data_j_ptr[d];
				}
				f_norm = 1.0f / f_norm;
				distance_data[index] += sqrt(f_norm);
				for (int d = 0; d < dim_; d++)
				{
					component_force_data_j_ptr[d] = component_force_data_j_ptr[d] * f_norm;
				}
			}
			else
			{
				memset(component_force_data_j_ptr, 0, dim_ * sizeof(float));
			}
		}
	}

	__global__ void calccombineforce_kernel(int electorn_num_, int dim_, int x, 
		const double* position_data_i_ptr, double* component_force_data, double* distance_data)
	{
		CUDA_KERNEL_LOOP(index, electorn_num_)
		{
			double* component_force_data_j_ptr = component_force_data + dim_ * index;
			if (x != index)
			{
				double f_norm = 0.0f;
				for (int d = 0; d < dim_; d++)
				{
					component_force_data_j_ptr[d] = component_force_data_j_ptr[d] - position_data_i_ptr[d];
					f_norm += component_force_data_j_ptr[d] * component_force_data_j_ptr[d];
				}
				f_norm = 1.0f / f_norm;
				distance_data[index] += sqrt(f_norm);
				for (int d = 0; d < dim_; d++)
				{
					component_force_data_j_ptr[d] = component_force_data_j_ptr[d] * f_norm;
				}
			}
			else
			{
				memset(component_force_data_j_ptr, 0, dim_ * sizeof(double));
			}
		}
	}

	__global__ void gemv_kernel(int electorn_num_, int dim_, 
		double* component_force_data, double* combine_force_data_i_ptr)
	{
		CUDA_KERNEL_LOOP(index, dim_)
		{
			for (int j = 0; j < electorn_num_; j++)
			{
				combine_force_data_i_ptr[index] += component_force_data[j * dim_ + index];
			}
		}
	}

	template<>
	void plasma<float>::calccombineforce_gpu(const float* position_data, float* combine_force_data,
		float* component_force_data, float* distance_data)
	{
		cudaSetDevice(device_);
		const float* multiplier_data = multiplier_->gpu_data();
		const float alpha = 1.0;
		const float beta = 0.0;
		for (int i = 0; i < electorn_num_; i++)
		{
			profiler->ScopeStart("copy");
			CUDA_CHECK(cudaMemcpy(component_force_data, position_data, electorn_num_ * dim_ * sizeof(float),
				cudaMemcpyKind::cudaMemcpyDefault));
			profiler->ScopeEnd();
			float* combine_force_data_i_ptr = combine_force_data + dim_ * i;
			const float* position_data_i_ptr = position_data + dim_ * i;
			profiler->ScopeStart("kernel");
			calccombineforce_kernel << <glasssix::excalibur::CUDA_GET_BLOCKS(electorn_num_), glasssix::excalibur::CUDA_NUM_THREADS >> >
				(electorn_num_, dim_, i, position_data_i_ptr, component_force_data, distance_data);
			CUDA_POST_KERNEL_CHECK;
			profiler->ScopeEnd();
			profiler->ScopeStart("gemv");
			CUBLAS_CHECK(cublasSgemv(cublas_handle_, cublasOperation_t::CUBLAS_OP_N, dim_, electorn_num_, &alpha,
				component_force_data, dim_, multiplier_data, 1, &beta, combine_force_data_i_ptr, 1));
			profiler->ScopeEnd();
		}
	}

	template<>
	void plasma<double>::calccombineforce_gpu(const double* position_data, double* combine_force_data,
		double* component_force_data, double* distance_data)
	{
		cudaSetDevice(device_);
		const double* multiplier_data = multiplier_->gpu_data();
		const double alpha = 1.0;
		const double beta = 0.0;
		for (int i = 0; i < electorn_num_; i++)
		{
			profiler->ScopeStart("copy");
			CUDA_CHECK(cudaMemcpy(component_force_data, position_data, electorn_num_ * dim_ * sizeof(double),
				cudaMemcpyKind::cudaMemcpyDefault));
			profiler->ScopeEnd();
			double* combine_force_data_i_ptr = combine_force_data + dim_ * i;
			const double* position_data_i_ptr = position_data + dim_ * i;
			profiler->ScopeStart("kernel");
			calccombineforce_kernel << <glasssix::excalibur::CUDA_GET_BLOCKS(electorn_num_), glasssix::excalibur::CUDA_NUM_THREADS >> >
				(electorn_num_, dim_, i, position_data_i_ptr, component_force_data, distance_data);
			CUDA_POST_KERNEL_CHECK;
			profiler->ScopeEnd();
			profiler->ScopeStart("gemv");
			CUBLAS_CHECK(cublasDgemv(cublas_handle_, cublasOperation_t::CUBLAS_OP_N, dim_, electorn_num_, &alpha,
				component_force_data, dim_, multiplier_data, 1, &beta, combine_force_data_i_ptr, 1));
			/*gemv_kernel << <glasssix::excalibur::CUDA_GET_BLOCKS(electorn_num_), glasssix::excalibur::CUDA_NUM_THREADS >> >
				(electorn_num_, dim_, component_force_data, combine_force_data_i_ptr);*/
			profiler->ScopeEnd();
		}
	}
#endif
}