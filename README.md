# ThomsonProblemSolver
This repo is a simple solver for n(n>=2) dimension [Thomson Problem](https://en.wikipedia.org/wiki/Thomson_problem).

## Requirements
- GLOG(depends on GFlags)
- OpenBLAS/Intel MKL
- optional: CUDA/cuBLAS

## Update
We optimized the implementation on CPU and fixed bugs on GPU. It works 20 times faster than the code before.

## Usage
This solver is very easy to use.

    
	//init solver with default parameters
	//init electrons on the hypersphere(electrons number, dimension, device)
	thomson::solver<double> tps = thomson::solver<double>(470, 3, -1);
	tps.Random_Init_Electorns();
	//set solving parameters
	tps.set_base_lr_(1.0);
	tps.set_max_iter_(100000);
	tps.set_min_pe_error_(1e-6);
	tps.set_display_interval_(20);
	tps.set_fast_pe_calculation(true);
	tps.set_snapshot_interval_(10000);
	tps.set_lr_policy_(true);
	//solving start
	tps.Solve_Thomson_Problem(pls);


## Speed Performance
- *float* datatype cannot meet the accuracy requirements, all experiments are done on *double*.
- As we know, the double-precision floating point capability on NVIDIA GTX series sucks, we got even worse results in samll cases.

| Device | Number of Electrons | Dimension |   Time Cost pre Iter   |
| :-------: | :-------:| :------: | :------: |
| Intel i7-8700K | 470 | 3 | 7.27ms |
| NVIDIA GTX 1060 | 470 | 3 | 7.84ms |
| Intel i7-8700K | 10572 | 512 | 162816ms |
| NVIDIA GTX 1060 | 10572 | 512 | 82771ms |
| Intel Xeon E5-2673 v4 | 10572 | 512 | 193797ms |
| NVIDIA TESLA V100 | 10572 | 512 | 13812ms |

## Error Analysis
### 2 and 3 Dimension Case

We can check the results with the numerical solution(in 3 dimension) of the wiki page that we list above.

| Number of Electrons | Theoretical Potential Energy |   Solved Potential Energy   | Number of Iterations |
| :-------: | :-------:| :------: | :------: |
| 2 | 0.500000 | 0.500000 | 14 |
| 10 | 32.716949 | 32.717061 | 918 |
| 100 | 4448.350634 | 4448.415575 | 4586 |
| 470| 104822.886324 | 104823.779129 | 32930 |
| 972| 455651.080935 | 455653.838019 | 46433 |


### High Dimension Case

We need present a proposition:

**There are 2 randomly picked points on the $n(n\geq 2)$ dimension hyper-sphere of radius $R$. Let $L$ designate the Euclidean distance between these two points. The nature of the probability distribution of $L$ can be described as below:**

$$\rho_{L,n}(L)=\frac{L^{n-2}}{C(n)R^{n-1}}[1-(\frac{L}{2R})^2]^{\frac{n-3}{2}}$$

**The coefficients $C(n)$ are given as follows:**

$$C(n)=\sqrt{\pi}\frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}$$


## Known Issues
- ~~The GPU code should be completely rewritten. Now, with irrational structural design, GPU code even slower than CPU :( ...~~
- Some fast algorithms caused a certain degree of numerical precision error;