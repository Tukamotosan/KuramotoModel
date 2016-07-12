#include "Simulator.h"

Simulator::Simulator()
{
	w = 128; h = 128;
	N = w * h;
	K = 6 * N;
	rng(getTickCount());

	mueOmega = 0.0;
	sigmaOmega = 0.5;

	Thetas = Mat::zeros(w, h, CV_32F);
	ThetasDash = Mat::zeros(w, h, CV_32F);

	maxFilter = 2.0;
	sigmaFilter = 4.0;
	rFilter = get_r(sigmaFilter, maxFilter);

	beta = 1.2;
	R = 0.25;

	dt = 0.001;

	LapLapGaussFilter = make_filter(-20, 20, sigmaFilter, rFilter);

	cnt = 0;
}


Simulator::~Simulator()
{
}

/*!
*/
void Simulator::exec() {
#pragma omp parallel for
	for (int x1 = 0; x1 < w; x1++) {
#pragma omp parallel for
		for (int y1 = 0; y1 < h; y1++) {
			Mat m = Mat::zeros(w, h, CV_32F);
#pragma omp parallel for
			for (int x2 = -20; x2 <= 20; x2++) {
				float x3 = x1 + x2;
				if (x3 < 0 || x3 >= w) { continue; }
#pragma omp parallel for
				for (int y2 = -20; y2 <= 20; y2++) {
					float y3 = y1 + y2;
					if (y3 < 0 || y3 >= h) { continue; }
					float v = sin(Thetas.at<float>(x3, y3) - Thetas.at<float>(x1, y1));
					m.at<float>(x3, y3) = v;
				}
			}
			Mat m2;
			filter2D(m, m2, -1, LapLapGaussFilter);
			ThetasDash.at<float>(x1, y1) = (float)rng.gaussian(sigmaOmega) + K*m2.at<float>(x1, y1) / N;
		}
	}

	// compute Thetas
	Thetas += dt * ThetasDash;

	cnt++;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(Thetas, &minVal, &maxVal, &minLoc, &maxLoc);
	printf("cnt = %d\t(min,max)=(%.8f,%.8f)\n", cnt, minVal, maxVal);

	if ((cnt % 10) != 0) { return; }
	Mat gray_image(w, h, CV_8UC1);
	float max = 3.2;
	float v;
	
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			v = map(Thetas.at<float>(x, y), minVal, maxVal, 0, 255);
			gray_image.at<unsigned char>(y, x) = (unsigned char)v;
		}
	}
	imwrite("data\\chap3_2\\gray" + std::to_string(cnt) + ".jpg", gray_image);
}

/*!
*/
void Simulator::exec2() {
	/*Mat m0 = Mat::zeros(4, 4, CV_32F);
	m0.forEach<float>([](float &p, const int *position) -> void {
		p = position[0];
	});*/

#pragma omp parallel for
	for (int x1 = 0; x1 < w; x1++) {
#pragma omp parallel for
		for (int y1 = 0; y1 < h; y1++) {
			Mat m = Mat::zeros(w, h, CV_32F);
			float v = 0.0;
#pragma omp parallel for
			for (int x2 = -20; x2 <= 20; x2++) {
				float x3 = x1 + x2;
				if (x3 < 0 || x3 >= w) { continue; }
#pragma omp parallel for
				for (int y2 = -20; y2 <= 20; y2++) {
					float y3 = y1 + y2;
					if (y3 < 0 || y3 >= h) { continue; }
					//float v = sin(Thetas.at<float>(x3, y3) - Thetas.at<float>(x1, y1));
					//m.at<float>(x3, y3) = v;
					float theta = Thetas.at<float>(x3, y3) - Thetas.at<float>(x1, y1);
					float v = -sin(theta + beta) + R*sin(2.0*theta);
					m.at<float>(x3,y3) = v;
				}
			}
			Mat m2;
			filter2D(m, m2, -1, LapLapGaussFilter);
			ThetasDash.at<float>(x1, y1) = (float)rng.gaussian(sigmaOmega) + K*m2.at<float>(x1, y1) / N;
		}
	}

	// compute Thetas
	Thetas += dt * ThetasDash;

	cnt++;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(Thetas, &minVal, &maxVal, &minLoc, &maxLoc);
	printf("cnt = %d\t(min,max)=(%.8f,%.8f)\n", cnt, minVal, maxVal);

	if ((cnt % 10) != 0) { return; }
	Mat gray_image(w, h, CV_8UC1);
	float max = 3.2;
	float v;

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			v = map(Thetas.at<float>(x, y), minVal, maxVal, 0, 255);
			gray_image.at<unsigned char>(y, x) = (unsigned char)v;
		}
	}
	imwrite("data\\gray" + std::to_string(cnt) + ".jpg", gray_image);
}

/*!
* \brief make_filter. Make filter of fourth derivative of a Gaussian function.
* \param lmin
* \param lmax
* \param sigma
* \param r
* \return
*/
Mat Simulator::make_filter(const int lmin, const float lmax, const float sigma, const float r) {
	Mat filter = Mat::zeros(lmax - lmin + 1, lmax - lmin + 1, CV_32F);
	for (int x = lmin; x <= lmax; x++) {
		for (int y = lmin; y <= lmax; y++) {
			float v = laplap_gaussian((float)x, (float)y, sigma, r);
			filter.at<float>(x - lmin, y - lmin) = v;
		}
	}
	return filter;
}


/*!
* \brief laplap_gaussian. fourth derivative of a Gaussian function
* \param x
* \param y
* \param sigma
* \param r
* \return
*/
float Simulator::laplap_gaussian(const float x, const float y, const float sigma, const float r) {
	float sigma2 = sigma*sigma;
	float sigma4 = sigma2*sigma2;
	float sigma10 = sigma4*sigma4*sigma2;
	float a = 1.0 / (2.0*M_PI*sigma10);
	float b = x*x - 4.0*sigma2; b = b*b;
	float c = y*y - 4.0*sigma2; c = c*c;
	float d = 2.0*x*x*y*y;
	float e = -24.0*sigma4;
	float f = -(x*x + y*y) / (2.0*sigma2);
	return r*a*(b + c + d + e)*exp(f);
}

/*!
* \brief get_r
* \param sigma
* \param R
* \return
*/
float Simulator::get_r(const float sigma, const float R) {
	return R*M_PI*pow(sigma, 6.0f) / 4.0;
}

float Simulator::map(const float x, const float xMin, const float xMax, const float yMin, const float yMax) {
	return (x - xMin)*(yMax - yMin) / (xMax - xMin) + yMin;
}