//#pragma once
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_
#include "timer.hpp"
#include "plasma.hpp"

#define PropertyBuilderByName(type, name, access_permission)\
    access_permission:\
        type name;\
    public:\
    inline void set##name(type v) {\
        name = v;\
    }\
    inline type get##name() {\
        return name;\
    }\

namespace thomson
{
	template<typename Dtype>
	class solver
	{
		PropertyBuilderByName(int, max_iter_, private);
		PropertyBuilderByName(Dtype, min_pe_error_, private);
		PropertyBuilderByName(bool, lr_policy_, private);
		PropertyBuilderByName(bool, fast_pe_calculation, private);
		PropertyBuilderByName(int, display_interval_, private);
		PropertyBuilderByName(Dtype, base_lr_, private);

	public: 
		solver();

		void Solve_Thomson_Problem(plasma<Dtype>& pls);
	};
}

#endif // _SOLVER_HPP_