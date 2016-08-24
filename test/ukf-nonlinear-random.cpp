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
	float vel = (rand() % 1000000u) / (float)1e6 - 0.5f; 
	static float state = 0; 
	state += vel; 
	return state; //2 * theta + 1; 
}

float h_process(float x){
	return (x - 1) / 2.0f; 
}

Matrix<float, 3, 1> f_proc(const Matrix<float, 3, 1> &i, void *u){
	Matrix<float, 3, 1> data; 
	data << 
		i(0) + i(1), 
		i(1) + i(2),
		i(2); 
	return data; 
}

Matrix<float, 1, 1> h_proc(const Matrix<float, 3, 1> &i, void *u){
	Matrix <float, 1, 1> data; 
	data << i(0); 
	return data; 
}

Matrix<float, 3, 3> Q_discrete_white_noise_3(float dt, float var){
	Matrix<float, 3, 3> q; 
	q << 
		0.25f*pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2), 
		0.50f*pow(dt, 3), pow(dt, 2), dt, 
		0.50f*pow(dt, 2), dt, 1; 
	return q * var;  
}

int main(){
	UnscentedKalmanFilter<3, 1, 1> k(f_proc, h_proc, NULL); 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Matrix<float, 3, 1>((const float[]){0, 0.0, 0})); 

	// setup initial belief
	Matrix<float, 3, 3> P; 
	Matrix<float, 3, 3> Q; 
	Matrix<float, 1, 1> R; 
	P << 
		0.1, 0, 0, 
		0, 0.001, 0,
		0, 0, 10; 
	Q = Q_discrete_white_noise_3(1.0, 0.1);  
	R << pow(50.0f, 2); 

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
		float noisen = norm_dist(rng); 
		float noise = sin(rand() / M_PI); 
		float z[1] = {truth + noisen}; 
		if(rand() % 10 > 5) z[0] *= 0.0; 
		k.update(Matrix<float, 1, 1>(z)); 
		Matrix<float, 3, 1> xk = k.get_prediction(); 
		Matrix<float, 3, 3> P = k.get_P(); 
		printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", theta, truth, z[0], xk(0), xk(1), P(0, 0), P(0, 1), P(1, 0), P(1, 1)); 
		
		// start test after 10 iterations
		//if(it > 10) TEST(is_equal(truth, xk(0), 0.1f)); 
		it++; 
	}

	return 0; 
}
