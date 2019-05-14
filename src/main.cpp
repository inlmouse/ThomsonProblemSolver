#include "../include/solver.hpp"

int main(int argc, char* argv[])
{
	if (argc < 4 || argc > 5)
	{
		LOG(ERROR) << "Too many params.";
		return -1;
	}
	const int e_num = atoi(argv[1]);
	const int dim = atoi(argv[2]);
	const int device = atoi(argv[3]);
	//init solver with default parameters
	//init electrons on the hypersphere(electrons number, dimension, device)
	thomson::solver<double> tps = thomson::solver<double>(e_num, dim, device);
	if (argc == 5)
	{
		tps.Init_Electorns_From_File(std::string(argv[4]));
	}
	else // argc == 4
	{
		tps.Random_Init_Electorns();
	}
	//set solving parameters
	tps.set_base_lr_(1.0);
	tps.set_max_iter_(100000);
	//tps.set_min_pe_error_(1e-6);
	tps.set_display_interval_(20);
	tps.set_fast_pe_calculation(true);
	tps.set_lr_policy_(true);
	tps.set_snapshot_interval_(10000);
	//solving start
	tps.Solve_Thomson_Problem();
	return 0;
}