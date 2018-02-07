#include "../include/plasma.hpp"

int main()
{
	thomson::plasma<double> pls = thomson::plasma<double>(468, 3, -1);
	pls.StartingOptimization(true);
	return 0;
}