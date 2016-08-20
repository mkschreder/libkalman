#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include "KalmanFilter.hpp"

using namespace matrix; 

static inline bool is_equal(float a, float b, float e = FLT_EPSILON){
	return fabsf(a - b) < e; 
}

#define STR(x) #x
#define TEST(x) if(!(x)){ printf("test failed at %d, %s: %s\n", __LINE__, __FILE__, STR(x)); exit(-1); }

int main(){
	KalmanFilter<2, 1, 1> k; 
	const float dt = 0.01; 

	// set initial state
	k.set_xk(Vector<float, 2>((const float[2]){1, 120.0})); 

	// setup initial belief
	k.set_P(Matrix<float, 2, 2>((const float[2][2]){
		{ 1, 0 },
		{ 0, 100 }
	})); 

	// setup our state transition matrix with simple motion equation
	k.set_F(Matrix<float, 2, 2>((const float[2][2]){
		{ 1, dt }, 
		{ 0, 1 }
	})); 

	// setup measurement function
	k.set_H(Matrix<float, 1, 2>((const float[1][2]){{1, 0}})); 

	// setup measurement covariance 
	k.set_R(Matrix<float, 1, 1>((const float[1][1]){{5}})); 

	// setup process noise
	k.set_Q(Matrix<float, 2, 2>((const float[2][2]){
		{ 1, 0 }, 
		{ 0, 1 }
	})); 

	// process some test data 
	const float indata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 15, 16, 17, 18, 19}; 
	k.predict(Vector<float, 1>()); 
	k.update(Vector<float, 1>(&indata[0])); 
	
	for(size_t c = 0; c < sizeof(indata)/sizeof(indata[0]); c++){
		for(float j = 1.0f; j > 0; j -= dt){
			k.predict(Vector<float, 1>()); 
		}
		k.update(Vector<float, 1>(&indata[c])); 
		Vector<float, 2> xk = k.get_prediction(); 
		printf("[%f %f], [%f, %f]\n", indata[c], 1.0f, xk(0), xk(1)); 
	}

	Vector<float, 2> xk = k.get_prediction(); 
	TEST(is_equal(xk(0), 19.0f, 0.5f)); // position will be predicted past measurement 
	TEST(is_equal(xk(1), 1.0f, 0.05f)); 
}
