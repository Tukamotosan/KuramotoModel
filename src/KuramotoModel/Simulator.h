#pragma once
#include <opencv2/opencv.hpp>
#include <omp.h>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;

class Simulator
{
public:
	// 2D Matrix that contains all oscilator's frequency
	Mat Thetas;

	// Difference of Thetas
	Mat ThetasDash;

	// divergence (12)

	// Filter of fourth derivative of the Laplacian of the Gaussian
	Mat LapLapGaussFilter;

	// size of oscilators(n=w*h)
	int w, h;

	// size of oscilators and K
	int N, K;

	// paramters of filter
	float sigmaFilter, mueFilter, maxFilter, rFilter;

	int cnt;

	// generator of norm
	RNG rng;

	// mue and sigma of rng
	float sigmaOmega, mueOmega;

	// parameters of gamma(13)
	float beta, R;

	// delta time
	float dt;

	Simulator();
	~Simulator();

	/*!
	*/
	void exec();
	
	// Using eq.13 as PIF
	void exec2();

	/*!
	* \brief make_filter. Make filter of fourth derivative of a Gaussian function.
	* \param lmin
	* \param lmax
	* \param sigma
	* \param r
	* \return
	*/
	Mat make_filter(const int lmin, const float lmax, const float sigma, const float r);

	/*!
	* \brief laplap_gaussian. fourth derivative of a Gaussian function
	* \param x
	* \param y
	* \param sigma
	* \param r
	* \return
	*/
	float laplap_gaussian(const float x, const float y, const float sigma, const float r);

	/*!
	* \brief get_r
	* \param sigma
	* \param R
	* \return
	*/
	float get_r(const float sigma, const float R);

private:
	float map(const float x, const float xMin, const float xMax, const float yMin, const float yMax);
};

