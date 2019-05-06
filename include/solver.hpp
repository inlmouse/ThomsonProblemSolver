//#pragma once
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_
#include <glasssix\profiler.hpp>
#include "plasma.hpp"

#define PropertyBuilderByName(type, name, access_permission)\
    access_permission:\
        type name;\
    public:\
    inline void set_##name(type v) {\
        name = v;\
    }\
    inline type get_##name() {\
        return name;\
    }\

namespace thomson
{
	template<typename Dtype>
	class solver
	{
		glasssix::Profiler *profiler;

		PropertyBuilderByName(int, max_iter_, private);
		PropertyBuilderByName(Dtype, min_pe_error_, private);
		PropertyBuilderByName(bool, lr_policy_, private);
		PropertyBuilderByName(bool, fast_pe_calculation, private);
		PropertyBuilderByName(int, display_interval_, private);
		PropertyBuilderByName(Dtype, base_lr_, private);
	public: 
		solver();
		~solver();

		void Solve_Thomson_Problem(plasma<Dtype>& pls);
	};
}

#endif // _SOLVER_HPP_