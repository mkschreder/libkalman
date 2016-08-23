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

float f_process(float theta){
	return sin(theta); 
}

int main(){
	KalmanFilter<2, 1, 1> k; 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Matrix<float, 2, 1>((const float[]){1, 1.0})); 

	// setup initial belief
	Matrix<float, 2, 2> P; 
	Matrix<float, 1, 2> H; 
	Matrix<float, 2, 2> Q; 
	P <<
		1, 0,
		0, 100; 
	H << 
		1, 0; 
	Q <<
		0.001, 0.1,
		0.1, 0.01; 
	k.set_P(P); 

	// setup measurement function
	k.set_H(H); 

	// setup measurement covariance 
	k.set_R(Matrix<float, 1, 1>((const float[]){4})); 

	// setup process noise
	k.set_Q(Q); 

	// process some test data 
	for(float theta = 0; theta < M_PI * 5; theta += 0.05f){
		// setup our state transition matrix with simple motion equation
		Matrix<float, 2, 2> F; 
		F <<
			1, dt / 10.0f,
			0, 1; 
		k.set_F(F); 

		for(float th = 0; th < dt; th += (dt / 10.0f)){
			k.predict(Matrix<float, 1, 1>()); 
		}
		float truth = f_process(theta); 
		float noise = norm_dist(rng) * 0.1f; 
		float z[1] = {truth + noise}; 
		k.update(Matrix<float, 1, 1>(z)); 
		Matrix<float, 2, 1> xk = k.get_prediction(); 
		printf("%f, %f, %f, %f, %f\n", theta, truth, z[0], xk(0), xk(1)); 
	}

	Matrix<float, 2, 1> xk = k.get_prediction(); 
}
