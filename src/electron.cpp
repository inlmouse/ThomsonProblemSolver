#include "../include/electron.hpp"
#include <random>
#include <cblas.h>
#include <iostream>

namespace thomson
{
	template <typename Dtype>
	electron<Dtype>::electron(int dimension, int device)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> d(0, 1);
		dimension_ = dimension;
		device_ = device;
		position_ = new tensor<Dtype>(std::vector<int>{dimension_}, device_);
		combine_force_ = new tensor<Dtype>(std::vector<int>{dimension_}, device_);
		memset(combine_force_->mutable_cpu_data(), 0, dimension * sizeof(Dtype));
		Dtype* position_data = position_->mutable_cpu_data();
		for (int i = 0; i < dimension; i++)
		{
			position_data[i] = static_cast<Dtype>(d(gen));
		}
		normalize2shpere_cpu();
	}

	template <typename Dtype>
	electron<Dtype>::~electron()
	{
		delete position_;
		delete combine_force_;
	}

	template <typename Dtype>
	electron<Dtype>::electron(const electron<Dtype>& e)
	{
		dimension_ = e.dimension_;
		device_ = e.device_;
		position_ = new tensor<Dtype>(std::vector<int>{e.dimension_}, e.device_);
		combine_force_ = new tensor<Dtype>(std::vector<int>{e.dimension_}, e.device_);
		memcpy(position_->mutable_cpu_data(), e.position_->cpu_data(), dimension_);
		memcpy(combine_force_->mutable_cpu_data(), e.combine_force_->cpu_data(), dimension_);
	}

	template <typename Dtype>
	const electron<Dtype>& electron<Dtype>::operator=(const electron<Dtype>& e)
	{
		if (position_ != nullptr)
		{
			delete position_;
			position_ = nullptr;
		}
		if (combine_force_ != nullptr)
		{
			delete combine_force_;
			combine_force_ = nullptr;
		}
		dimension_ = e.dimension_;
		device_ = e.device_;
		position_ = new tensor<Dtype>(std::vector<int>{e.dimension_}, e.device_);
		combine_force_ = new tensor<Dtype>(std::vector<int>{e.dimension_}, e.device_);
		memcpy(position_->mutable_cpu_data(), e.position_->cpu_data(), dimension_);
		memcpy(combine_force_->mutable_cpu_data(), e.combine_force_->cpu_data(), dimension_);
		return *this;
	}

	template<>
	void electron<float>::normalize2shpere_cpu()
	{
		float norm = cblas_snrm2(dimension_, position_->cpu_data(), 1);
		cblas_sscal(dimension_, 1 / norm, position_->mutable_cpu_data(), 1);
	}

	template<>
	void electron<double>::normalize2shpere_cpu()
	{
		double norm = cblas_dnrm2(dimension_, position_->cpu_data(), 1);
		cblas_dscal(dimension_, 1 / norm, position_->mutable_cpu_data(), 1);
	}

	template <typename Dtype>
	void electron<Dtype>::combineforce2zero_cpu()
	{
		memset(combine_force_->mutable_cpu_data(), 0, dimension_ * sizeof(Dtype));
	}

	template <>
	void electron<float>::add1componentforce_cpu(const float* other_position)
	{
		float* f_i = new float[dimension_];
		memcpy(f_i, other_position, dimension_ * sizeof(float));
		// f_i = \frac{x_i - x_j}{\|x_i - x_j\|_2^2}
		cblas_saxpby(dimension_, 1.0, position_->cpu_data(), 1, -1.0, f_i, 1);
		float f_norm = cblas_snrm2(dimension_, f_i, 1);
		cblas_sscal(dimension_, 1 / (f_norm*f_norm), f_i, 1);
		// f = f + f_i
		cblas_saxpy(dimension_, 1.0, f_i, 1, combine_force_->mutable_cpu_data(), 1);
		//
		delete f_i;
	}

	template <>
	void electron<double>::add1componentforce_cpu(const double* other_position)
	{
		double* f_i = new double[dimension_];
		memcpy(f_i, other_position, dimension_ * sizeof(double));
		// f_i = \frac{x_i - x_j}{\|x_i - x_j\|_2^2}
		cblas_daxpby(dimension_, 1.0, position_->cpu_data(), 1, -1.0, f_i, 1);
		double f_norm = cblas_dnrm2(dimension_, f_i, 1);
		cblas_dscal(dimension_, 1 / (f_norm*f_norm), f_i, 1);
		// f = f + f_i
		cblas_daxpy(dimension_, 1.0, f_i, 1, combine_force_->mutable_cpu_data(), 1);
		//
		delete f_i;
	}

	template <>
	float electron<float>::calculatedistance_cpu(const float* other_position)
	{
		float* f_i = new float[dimension_];
		memcpy(f_i, other_position, dimension_ * sizeof(float));
		// f_i = \frac{x_i - x_j}{\|x_i - x_j\|_2^2}
		cblas_saxpby(dimension_, 1.0, position_->cpu_data(), 1, -1.0, f_i, 1);
		float f_norm = cblas_snrm2(dimension_, f_i, 1);
		delete f_i;
		return f_norm;
	}

	template <>
	double electron<double>::calculatedistance_cpu(const double* other_position)
	{
		double* f_i = new double[dimension_];
		memcpy(f_i, other_position, dimension_ * sizeof(double));
		// f_i = \frac{x_i - x_j}{\|x_i - x_j\|_2^2}
		cblas_daxpby(dimension_, 1.0, position_->cpu_data(), 1, -1.0, f_i, 1);
		double f_norm = cblas_dnrm2(dimension_, f_i, 1);
		delete f_i;
		return f_norm;
	}

	template <>
	void electron<float>::updateposition_cpu(float lr)
	{
		cblas_saxpy(dimension_, 1.0*lr, combine_force_->cpu_data(), 1, position_->mutable_cpu_data(), 1);
		normalize2shpere_cpu();
		//
		memset(combine_force_->mutable_cpu_data(), 0, dimension_ * sizeof(float));
	}

	template <>
	void electron<double>::updateposition_cpu(double lr)
	{
		cblas_daxpy(dimension_, 1.0*lr, combine_force_->cpu_data(), 1, position_->mutable_cpu_data(), 1);
		normalize2shpere_cpu();
		//
		memset(combine_force_->mutable_cpu_data(), 0, dimension_ * sizeof(double));
	}

	template <typename Dtype>
	const tensor<Dtype>* electron<Dtype>::getcurrentposition()
	{
		return position_;
	}

	template class electron<float>;
	template class electron<double>;
}