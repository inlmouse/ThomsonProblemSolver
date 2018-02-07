#include "../include/solver.hpp"

int main()
{
	thomson::plasma<double> pls = thomson::plasma<double>(468, 3, -1);
	thomson::solver<double> tps = thomson::solver<double>();
	tps.Solve_Thomson_Problem(pls);
	return 0;
}