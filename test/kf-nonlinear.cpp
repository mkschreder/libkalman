#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <random>
#include <chrono>

#include "KalmanFilter.hpp"
#include "test-macros.hpp"

using namespace matrix; 

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
	k.set_xk(Vector<float, 2>((const float[2]){1, 1.0})); 

	// setup initial belief
	k.set_P(Matrix<float, 2, 2>((const float[2][2]){
		{ 1, 0 },
		{ 0, 100 }
	})); 

	// setup measurement function
	k.set_H(Matrix<float, 1, 2>((const float[1][2]){{1, 0}})); 

	// setup measurement covariance 
	k.set_R(Matrix<float, 1, 1>((const float[1][1]){{4}})); 

	// setup process noise
	k.set_Q(Matrix<float, 2, 2>((const float[2][2]){
		{ 0.001, 0.1 }, 
		{ 0.1, 0.01 }
	})); 

	// process some test data 
	for(float theta = 0; theta < M_PI * 5; theta += 0.05f){
		// setup our state transition matrix with simple motion equation
		k.set_F(Matrix<float, 2, 2>((const float[2][2]){
			{ 1, dt / 10.0f }, 
			{ 0, 1 }
		})); 
		for(float th = 0; th < dt; th += (dt / 10.0f)){
			k.predict(Vector<float, 1>()); 
		}
		float truth = f_process(theta); 
		float noise = norm_dist(rng) * 0.1f; 
		float z[1] = {truth + noise}; 
		k.update(Vector<float, 1>(z)); 
		Vector<float, 2> xk = k.get_prediction(); 
		printf("%f, %f, %f, %f, %f\n", theta, truth, z[0], xk(0), xk(1)); 
	}

	Vector<float, 2> xk = k.get_prediction(); 
}
