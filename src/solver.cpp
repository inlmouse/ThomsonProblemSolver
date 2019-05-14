#include "../include/solver.hpp"
#include <iomanip>

namespace thomson
{
	template <typename Dtype>
	solver<Dtype>::solver(int e_num, int dim, int device)
	{
		electorn_num_ = e_num;
		dim_ = dim;
		device_ = device;
		CHECK_GE(electorn_num_, dim_);
		pls_ = new plasma<Dtype>(electorn_num_, dim_, device_);
		max_iter_ = std::numeric_limits<int>::max();
		min_pe_error_ = std::numeric_limits<Dtype>::min();
		lr_policy_ = true;
		fast_pe_calculation = true;
		display_interval_ = 50;
		base_lr_ = (Dtype)1.0;
		snapshot_interval_ = max_iter_ / 10;
		//profiler = glasssix:://profiler::Get();
		//profiler->TurnON();
	}

	template <typename Dtype>
	solver<Dtype>::~solver()
	{
		//profiler->TurnOFF();
		std::string filename = "tps.json";
		//profiler->DumpProfile(filename.c_str());
		if (max_iter_ % snapshot_interval_ != 0)
		{
			pls_->Dump_Electorns_To_File("tps_" + std::to_string(electorn_num_) +
				"_" + std::to_string(dim_) + "_" + typeid(Dtype).name()
				+ "_iter" + std::to_string(max_iter_) + ".dat");
		}
		delete pls_;
	}

	template<typename Dtype>
	void solver<Dtype>::Random_Init_Electorns()
	{
		pls_->Random_Init_Electorns();
	}

	template<typename Dtype>
	void solver<Dtype>::Init_Electorns_From_File(std::string path)
	{
		pls_->Init_Electorns_From_File(path);
	}

	template <typename Dtype>
	void solver<Dtype>::Solve_Thomson_Problem()
	{
		CHECK_GE(base_lr_, 0);
		if (device_ < 0)
		{
			LOG(INFO) << "Working on CPU Device: " << device_;
			int num_thread;
#ifdef _OPENMP
			num_thread = omp_get_num_procs();
			omp_set_num_threads(num_thread);
			LOG(INFO) << "There are " << num_thread << " threads, " << omp_get_num_procs() << " processors.";
#else
			num_thread = 1;
#endif
		}
		else
		{
			LOG(INFO) << "Working on GPU Device: " << device_;
			/*cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;*/
		}
		LOG(INFO) << "Number of Electorns: " << electorn_num_;
		LOG(INFO) << "Number of Dimension: " << dim_;
		LOG(INFO) << "Max iter: "<< max_iter_;
		LOG(INFO) << "Min Potential Energy Error: "<< min_pe_error_;
		LOG(INFO) << "Learning Rate Policy(automatic scaled to accelerate solving or not): "<< (lr_policy_?"True":"False");
		LOG(INFO) << "Iters Interval for Displaying: "<< display_interval_;
		LOG(INFO) << "Base Learning Rate: "<< base_lr_;
		LOG(INFO) << "Solving start...";
		//
		int counter = 0;
		Dtype lr = base_lr_;
		while (counter < max_iter_)
		{
			//profiler->ScopeStart("Forward");
			pls_->Forward();
			//profiler->ScopeEnd();
			//profiler->ScopeStart("Backward");
			pls_->Backward(lr);
			//profiler->ScopeEnd();
			
			Dtype temp_E;
			temp_E = pls_->CalculatePotentialEnergy();

			//fast learning rate policy
			Dtype error = abs(temp_E - pls_->GetPE());
			if (lr_policy_)
			{
				Dtype log_error = log(error);
				if (log_error<(Dtype)0)
				{
					lr = -1 * log_error;
				}
			}
			std::stringstream ss;
			ss << std::fixed << std::setprecision(6) << temp_E;
			LOG_IF(INFO, counter % display_interval_ == 0) << "Iter " << counter << ": PotentialEnergy = " << ss.str();

			if (error < min_pe_error_)
			{
				LOG(INFO) << "Iter " << counter << " finished! Final PotentialEnergy = " << ss.str();
				break;
			}
			else
			{
				pls_->SetPE(temp_E);
			}
			counter++;
			if (counter % snapshot_interval_ == 0)
			{
				pls_->Dump_Electorns_To_File("tps_" + std::to_string(electorn_num_) +
					"_" + std::to_string(dim_) + "_" + typeid(Dtype).name()
					+"_iter" + std::to_string(counter)+ ".dat");
			}
		}
	}

	template class solver<float>;
	template class solver<double>;
}