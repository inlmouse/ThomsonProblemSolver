#include "../include/solver.hpp"

int main()
{
	//init electrons on the hypersphere(electrons number, dimension, device)
	thomson::plasma<double> pls = 
		thomson::plasma<double>(10572, 512, -1);
	pls.Random_Init_Electorns();
	//init solver with default parameters
	thomson::solver<double> tps = thomson::solver<double>();
	//set solving parameters
	tps.set_base_lr_(1.0);
	tps.set_max_iter_(2);
	//tps.set_min_pe_error_(1e-6);
	tps.set_display_interval_(1);
	tps.set_fast_pe_calculation(true);
	tps.set_lr_policy_(true);
	//solving start
	tps.Solve_Thomson_Problem(pls);
	return 0;
}