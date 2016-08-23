#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include "KalmanFilter.hpp"
#include "test-macros.hpp"

using namespace Eigen; 
using namespace filter; 

int main(){
	KalmanFilter<2, 1, 1> k; 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Matrix<float, 2, 1>((const float[]){1, 120.0})); 

	// setup initial belief
	Matrix<float, 2, 2> P; 
	P <<
		1, 0,
		0, 100; 
	k.set_P(P); 

	// setup our state transition matrix with simple motion equation
	Matrix<float, 2, 2> F;
	F << 
		  1, dt , 
		  0, 1; ; 
	k.set_F(F); 

	// setup measurement function
	k.set_H(Matrix<float, 1, 2>((const float[]){1, 0})); 

	// setup measurement covariance 
	k.set_R(Matrix<float, 1, 1>((const float[]){5})); 

	// setup process noise
	Matrix<float, 2, 2> Q; 
	Q <<
		1, 0,
		0, 1; 
	k.set_Q(Q); 

	// process some test data 
	const float indata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 15, 16, 17, 18, 19}; 
	k.predict(Matrix<float, 1, 1>()); 
	k.update(Matrix<float, 1, 1>(&indata[0])); 

	printf("F: %f %f\n", F(0, 1), F(1, 0));  
	for(size_t c = 0; c < sizeof(indata)/sizeof(indata[0]); c++){
		for(float j = 1.0f; j > 0; j -= dt){
			k.predict(Matrix<float, 1, 1>()); 
		}
		k.update(Matrix<float, 1, 1>(&indata[c])); 
		Matrix<float, 2, 1> xk = k.get_prediction(); 
		printf("[%f %f], [%f, %f]\n", indata[c], 1.0f, xk(0), xk(1)); 
	}

	Matrix<float, 2, 1> xk = k.get_prediction(); 
	TEST(is_equal(xk(0), 19.0f, 0.5f)); // position will be predicted past measurement 
	TEST(is_equal(xk(1), 1.0f, 0.05f)); 
}
