#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <random>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

#include "KalmanFilter.hpp"
#include "ConstantVelocityPositionFilter.hpp"
#include "test-macros.hpp"

using namespace std; 
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

Matrix<float, 2, 2> Q_discrete_white_noise_2(float dt, float var){
	Matrix<float, 2, 2> q; 
	q << 
		0.25f*pow(dt, 4), 0.5 * pow(dt, 3), 
		0.50f*pow(dt, 3), pow(dt, 2); 
	return q * var;  
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
	ConstantVelocityPositionFilter<float> k(40, 40.0f, 1.0f); 
	Eigen::Matrix<float, 2, 1> xk; 
	xk << 1.5, 0; 
	k.set_prediction(xk); 

	int it = 0; 

	std::ifstream file("rangefinder_data.csv"); 
	string line; 
	if(!file.is_open()) {
		printf("Could not open sample file!\n"); 
		exit(-1); 
	}

	unsigned int i = 0; 

	while(std::getline(file, line)){
		float front, back, right, left; 
		sscanf(line.c_str(), "%f, %f, %f, %f", &front, &back, &right, &left); 
		
		k.predict(); 
		//if(front > 1.5) front = 2.0f; 
		if(front > 0.3f)
			k.input_position(front); 

		Matrix<float, 2, 1> xk = k.get_prediction(); 
		printf("%f, %f, %f, %f, %f\n", (float)it, 0.0f, front, xk(0), xk(1)); 
		
		// start test after 10 iterations
		//if(it > 10) TEST(is_equal(truth, xk(0), 0.1f)); 
		it++; 
	}

	return 0; 
}
