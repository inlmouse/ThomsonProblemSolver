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
		//glasssix::Profiler *profiler;
		plasma<Dtype> *pls_;

		int electorn_num_;
		int dim_;
		int device_;


		PropertyBuilderByName(int, max_iter_, private);
		PropertyBuilderByName(Dtype, min_pe_error_, private);
		PropertyBuilderByName(bool, lr_policy_, private);
		PropertyBuilderByName(bool, fast_pe_calculation, private);
		PropertyBuilderByName(int, display_interval_, private);
		PropertyBuilderByName(Dtype, base_lr_, private);
		PropertyBuilderByName(int, snapshot_interval_, private);
	public:
		solver() = delete;
		solver(int e_num, int dim, int device);
		~solver();

		void Random_Init_Electorns();
		void Init_Electorns_From_File(std::string path);
		void Solve_Thomson_Problem();
	};
}

#endif // _SOLVER_HPP_