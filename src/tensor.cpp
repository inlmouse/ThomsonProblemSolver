#include "../include/tensor.hpp"

namespace thomson
{
	template <typename Dtype>
	tensor<Dtype>::tensor(const std::vector<int>& shape, int device)
	{
		count_ = 1;
		for (int i = 0; i < shape.size(); i++)
		{
			count_ *= shape[i];
			shape_.push_back(shape[i]);
		}
		device_ = device;
		data_ = new syncedmem(count_ * sizeof(Dtype), device_);
	}

	template <typename Dtype>
	tensor<Dtype>::~tensor()
	{
		delete data_;
	}

	template<typename Dtype>
	const Dtype* tensor<Dtype>::cpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->cpu_data());
	}

	template<typename Dtype>
	const Dtype* tensor<Dtype>::gpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->gpu_data());
	}

	template<typename Dtype>
	Dtype* tensor<Dtype>::mutable_cpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_cpu_data());
	}

	template<typename Dtype>
	Dtype* tensor<Dtype>::mutable_gpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_gpu_data());
	}

	template<typename Dtype>
	void tensor<Dtype>::set_cpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
		}
		data_->set_cpu_data(data);
	}

	template<typename Dtype>
	void tensor<Dtype>::set_gpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
		}
		data_->set_gpu_data(data);
	}

	template class tensor<float>;
	template class tensor<double>;
}