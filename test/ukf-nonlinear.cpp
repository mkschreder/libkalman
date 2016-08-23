#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <random>
#include <chrono>

#include "KalmanFilter.hpp"
#include "test-macros.hpp"

using namespace Eigen; 
using namespace filter; 

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine rng(seed);
std::normal_distribution<double> norm_dist(0.0,1.0);

#define limit(x, a, b) (((x) < (a))?(a):(((x) > (b))?(b):(x)))

float f_process(float theta){
	return sin(theta) * 10; //2 * theta + 1; 
}

float h_process(float x){
	return (x - 1) / 2.0f; 
}

Matrix<float, 2, 1> f_proc(const Matrix<float, 2, 1> &i, void *u){
	Matrix<float, 2, 1> data; 
	data << 
		i(0) + i(1), 
		i(1); 
	return data; 
}

Matrix<float, 1, 1> h_proc(const Matrix<float, 2, 1> &i, void *u){
	Matrix <float, 1, 1> data; 
	data << i(0); 
	return data; 
}

int main(){
	UnscentedKalmanFilter<2, 1, 1> k(f_proc, h_proc, NULL); 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Matrix<float, 2, 1>((const float[]){0, 0.0})); 

	// setup initial belief
	Matrix<float, 2, 2> P; 
	Matrix<float, 2, 2> Q; 
	Matrix<float, 1, 1> R; 
	P << 
		0.1, 0, 
		0, 0.001; 
	Q << 
		0.001, 0.03,
		0.03, 0.01; 
	R << 1.0f; 

	k.set_P(P); 

	// setup measurement covariance 
	k.set_R(R); 

	// setup process noise
	k.set_Q(Q); 

	// process some test data 
	int it = 0; 
	for(float theta = 0; theta < M_PI * 5; theta += 0.01f){
		//for(float th = 0; th < dt; th += (dt / 10.0f)){
			k.predict(Matrix<float, 1, 1>()); 
		//}
		float truth = f_process(theta); 
		float noise = norm_dist(rng); 
		float z[1] = {truth + noise}; 
		k.update(Matrix<float, 1, 1>(z)); 
		Matrix<float, 2, 1> xk = k.get_prediction(); 
		printf("%f, %f, %f, %f, %f\n", theta, truth, z[0], xk(0), xk(1)); 
		
		// start test after 10 iterations
		//if(it > 10) TEST(is_equal(truth, xk(0), 0.1f)); 
		it++; 
	}

	return 0; 
}
