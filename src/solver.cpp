#include "../include/solver.hpp"
#include <iomanip>

namespace thomson
{
	template <typename Dtype>
	solver<Dtype>::solver()
	{
		max_iter_ = std::numeric_limits<int>::max();
		min_pe_error_ = std::numeric_limits<Dtype>::min();
		lr_policy_ = true;
		fast_pe_calculation = true;
		display_interval_ = 50;
		base_lr_ = (Dtype)1.0;
		profiler = glasssix::Profiler::Get();
		profiler->TurnON();
	}

	template <typename Dtype>
	solver<Dtype>::~solver()
	{
		profiler->TurnOFF();
		std::string filename = "tps.json";
		profiler->DumpProfile(filename.c_str());
	}

	template <typename Dtype>
	void solver<Dtype>::Solve_Thomson_Problem(plasma<Dtype>& pls)
	{
		int num_thread;
#ifdef _OPENMP
		num_thread = omp_get_num_procs();
		omp_set_num_threads(num_thread);
		LOG(INFO) << "There are " << num_thread << " threads, " << omp_get_num_procs() << " processors.";
#else
		num_thread = 1;
#endif
		LOG(INFO) << "Solving start...";
		LOG(INFO) << "Max iter: "<< max_iter_;
		LOG(INFO) << "Min Potential Energy Error: "<< min_pe_error_;
		LOG(INFO) << "Learning Rate Policy(automatic scaled to accelerate solving or not): "<< (lr_policy_?"True":"False");
		LOG(INFO) << "Fast Plasma Potential Energy Calculation(only works with OpenMP): "<< (fast_pe_calculation ? "True" : "False");
		LOG(INFO) << "Iters Interval for Displaying: "<< display_interval_;
		LOG(INFO) << "Base Learning Rate: "<< base_lr_;
		//
		int counter = 0;
		Dtype lr = base_lr_;
		while (counter < max_iter_)
		{
			profiler->ScopeStart("Forward");
			pls.Forward();
			profiler->ScopeEnd();
			profiler->ScopeStart("Backward");
			pls.Backward(lr);
			profiler->ScopeEnd();
			
			Dtype temp_E;
			temp_E = pls.CalculatePotentialEnergy();

			//fast learning rate policy
			Dtype error = abs(temp_E - pls.GetPE());
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
				pls.SetPE(temp_E);
			}
			counter++;
		}
	}

	template class solver<float>;
	template class solver<double>;
}