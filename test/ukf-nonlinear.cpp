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
	return sin(theta) * 10; //2 * theta + 1; 
}

float h_process(float x){
	return (x - 1) / 2.0f; 
}

matrix::Vector<float, 2> f_proc(const matrix::Vector<float, 2> &i){
	float data[2] = {i(0) + i(1), i(1)}; 
	return matrix::Vector<float, 2>(data); 
}

matrix::Vector<float, 1> h_proc(const matrix::Vector<float, 2> &i){
	float data[1] = {i(0)}; //{h_process(i(0))}; 
	return matrix::Vector<float, 1>(data); 
}

int main(){
	UnscentedKalmanFilter<2, 1, 1> k(f_proc, h_proc); 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Vector<float, 2>((const float[2]){1, 1.0})); 

	// setup initial belief
	k.set_P(Matrix<float, 2, 2>((const float[2][2]){
		{ 0.1, 0 },
		{ 0, 0.1 }
	})); 

	// setup measurement covariance 
	k.set_R(Matrix<float, 1, 1>((const float[1][1]){{2.0f}})); 

	// setup process noise
	k.set_Q(Matrix<float, 2, 2>((const float[2][2]){
		{ 5.0, 0.0 }, 
		{ 0.0, 1.0 }
	})); 

	// process some test data 
	int it = 0; 
	for(float theta = 0; theta < M_PI * 5; theta += 0.05f){
		for(float th = 0; th < dt; th += (dt / 10.0f)){
			k.predict(Vector<float, 1>()); 
		}
		float truth = f_process(theta); 
		float noise = norm_dist(rng); 
		float z[1] = {truth + noise}; 
		k.update(Vector<float, 1>(z)); 
		Vector<float, 2> xk = k.get_prediction(); 
		printf("%f, %f, %f, %f, %f\n", theta, truth, z[0], xk(0), xk(1)); 
		
		// start test after 10 iterations
		//if(it > 10) TEST(is_equal(truth, xk(0), 0.1f)); 
		it++; 
	}

	return 0; 
}
