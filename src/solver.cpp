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
		//device_ = -1;
	}

	template <typename Dtype>
	void solver<Dtype>::Solve_Thomson_Problem(plasma<Dtype>& pls)
	{
		int counter = 0;
		Dtype lr = base_lr_;
		while (counter < max_iter_)
		{
			pls.CalculateAllForce();
			pls.UpdateAllPosition(lr);
			Dtype temp_E;
#ifdef _OPENMP
			if (fast_pe_calculation)
			{
				temp_E = pls.CalculatePotentialEnergy_Parallel();
			}
			else
			{
				temp_E = pls.CalculatePotentialEnergy();
			}
#else
			temp_E = pls.CalculatePotentialEnergy();
#endif
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