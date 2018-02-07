# ThomsonProblemSolver
This repo is a simple solver for n(n>=2) dimension [Thomson Problem](https://en.wikipedia.org/wiki/Thomson_problem).

## Requirements
- GLOG(depends on GFlags)
- OpenBLAS/Intel MKL
- optional: CUDA/cuBLAS

## Usage
This solver is very easy to use.

    //init electrons on the hypersphere(electrons number, dimension, device)
	thomson::plasma<double> pls = thomson::plasma<double>(470, 3, -1);
	//init solver with default parameters
	thomson::solver<double> tps = thomson::solver<double>();
	//set solving parameters
	tps.set_base_lr_(1.0);
	tps.set_max_iter_(100000);
	tps.set_min_pe_error_(1e-6);
	tps.set_display_interval_(20);
	tps.set_fast_pe_calculation(true);
	tps.set_lr_policy_(true);
	//solving start
	tps.Solve_Thomson_Problem(pls);


## Error Analysis
### 2 and 3 Dimension Case

We can check the results with the numerical solution(in 3 dimension) of the wiki page that we list above.

| Number of Electrons | Theoretical Potential Energy |   Solved Potential Energy   |
| :-------: | :-------:| :------: |
| 2 | 0.500000 | 0.500000 |
| 10 | 32.716949 | 32.717061 |
| 100 | 4448.350634 | 4448.415575 |
| 470| 104822.886324 | 104823.880431 |


### High Dimension

We need present a proposition:

**There are 2 randomly picked points on the $n(n\geq 2)$ dimension hyper-sphere of radius $R$. Let $L$ designate the Euclidean distance between these two points. The nature of the probability distribution of $L$ can be described as below:**

$$\rho_{L,n}(L)=\frac{L^{n-2}}{C(n)R^{n-1}}[1-(\frac{L}{2R})^2]^{\frac{n-3}{2}}$$

**The coefficients $C(n)$ are given as follows:**

$$C(n)=\sqrt{\pi}\frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}$$


## Known Issues
- The GPU code should be completely rewritten. Now, with irrational structural design, GPU code even slower than CPU :( ...
- Some fast algorithms caused a certain degree of numerical precision error;
