#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <random>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <semaphore.h>
#include <unistd.h>
#include <math.h>

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
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)


/*
float f_process(float theta){
	float vel = (rand() % 1000000u) / (float)1e6 - 0.5f; 
	static float state = 0; 
	state += vel; 
	return state; //2 * theta + 1; 
}

float h_process(float x){
	return (x - 1) / 2.0f; 
}
*/
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

#include <stdint.h>

struct velcurve {
	//float position_sp;
	//float velocity_sp;
	float target_acc;
	float target_vel;
	float target_dec;
	int8_t sign;
	struct {
		float x, y;
	} target;
	//! parts of the trapetzoid. Three points are reaching max vel, start of decel, stop.
	struct velcurve_part_def {
		//! duration of the part
		float time;
		//! distance traveled during this part
		float distance;
		//! acceleration/deceleration of this part
		float accel;
		//! intial velocity when entering the part
		float initial_velocity;
		//! final velocity after ending this part
		float final_velocity;
	} parts[3];
};

void velcurve_init(struct velcurve *self, float max_acc, float max_speed, float max_dec);
void velcurve_plan_move(struct velcurve *self, float distance, float vel_start, float vel_end);
int velcurve_get_desired_pv_for_t(struct velcurve *self, float ts, float *pos_sp, float *vel_sp);
void velcurve_init(struct velcurve *self, float max_acc, float max_speed, float max_dec){
	self->target_acc = max_acc;
	self->target_vel = max_speed;
	self->target_dec = max_dec;
	self->sign = 1;
}

void velcurve_plan_move(struct velcurve *self, float distance, float vel_start, float vel_end){
	// use positions in meters
/*
	float p_start = self->state.wheel[0].pos_mm * 0.001f;
	float p_end = p_start + (self->current_move.target.x * 0.001f);

	float self->target_acc = 0.5;
	float self->target_vel = 1.0;
	float self->target_dec = -0.5;
*/

	if(distance < 0) {
		distance = -distance;
		self->sign = -1;
	} else {
		self->sign = 1;
	}

	float t1, t2, t3, v1;
	float d1 = pow(self->target_vel, 2) - pow(vel_start, 2) / (2 * self->target_acc);
	float d2 = 0;
	float d3 = pow(vel_end, 2) - pow(self->target_vel, 2) / (2 * self->target_dec);

	if(d1 + d3 <= distance){
		t1 = (self->target_vel - vel_start) / self->target_acc;
		t3 = (vel_end - self->target_vel) / self->target_dec;

		d2 = distance - d1 - d3;
		t2 = d2 / self->target_vel;

		v1 = self->target_vel;
	} else {
		d1 = (2 * self->target_dec * distance + pow(vel_start, 2) - pow(vel_end, 2)) / ( 2 * (self->target_dec - self->target_acc));
		d2 = 0;
		d3 = distance - d1;

		t1 = MAX(sqrt(2 * self->target_acc * d1 + pow(vel_start, 2)) - vel_start, 0) / self->target_acc;
		v1 = vel_start + self->target_acc * t1;

		t2 = 0;
		t3 = (vel_end - v1) / self->target_dec;
	}

	float times[] = { t1, t2, t3 };
	float dists[] = { d1, d2, d3 };
	float accels[] = { self->target_acc, 0, self->target_dec };

	struct velcurve::velcurve_part_def *p = self->parts;
	for(int c = 0; c < 3; c++){
		p[c].time = times[c];
		p[c].distance = dists[c];
		p[c].accel = accels[c];
	}
	p[0].initial_velocity = vel_start;
	p[0].final_velocity = v1;
	p[1].initial_velocity = v1;
	p[1].final_velocity = v1;
	p[2].initial_velocity = v1;
	p[2].final_velocity = vel_end;
}

int velcurve_get_desired_pv_for_t(struct velcurve *self, float ts, float *pos_sp, float *vel_sp){
	float time = 0;
	float distance = 0;
	float velocity = 0;
	int done = 0;
	for(int c = 0; c < 3; c++){
		struct velcurve::velcurve_part_def *p = self->parts;
		if((time + p[c].time) < ts) {
			distance += p[c].distance;
			time += p[c].time;
			velocity = 0;
		} else {
			float t = ts - time;
			float d = p[c].initial_velocity * t + 0.5f * p[c].accel * pow(t, 2);
			velocity = sqrt(pow(p[c].initial_velocity, 2) + 2 * p[c].accel * d);
			distance += d;
			if(c == 2) done = 1;
			break;
		}
	}

	// update position setpoint
	*vel_sp = velocity * self->sign;
	*pos_sp = distance * self->sign;

	return done;
}

int main(){
	//sem_t _shutdown;
	//sem_init(&_shutdown, 0, 0);
	ConstantVelocityPositionFilter<float> k(0.01f, 0.01f, 0.001f); 
	ConstantVelocityPositionFilter<float> a(0.01f, 0.01f, 0.001f); 
	Eigen::Matrix<float, 3, 1> xk; 
	xk << 0, 0, 0; 
	k.set_prediction(xk); 
	a.set_prediction(xk); 

	int it = 0; 

/*
	std::ifstream file("rangefinder_data.csv"); 
	string line; 
	if(!file.is_open()) {
		printf("Could not open sample file!\n"); 
		exit(-1); 
	}
*/
	unsigned int i = 0; 

	printf("time = [];\n");
	printf("pos_x = [];\n");
	printf("vel_x = [];\n");
	printf("acc_x = [];\n");
	printf("pos_sp = [];\n");
	printf("vel_sp = [];\n");
	printf("int_pos = [];\n");
	//while(std::getline(file, line)){

	struct velcurve curve;
	velcurve_init(&curve, 0.5, 1.0f, -0.5);

	velcurve_plan_move(&curve, 3.0, 0, 0);
	float int_pos = 0;
	for(int c = 0; c < 6000; c++){
		//float front, back, right, left; 
		float pos_sp, vel_sp;
		velcurve_get_desired_pv_for_t(&curve, c * 1e-3, &pos_sp, &vel_sp);
		//sscanf(line.c_str(), "%f, %f, %f, %f", &front, &back, &right, &left); 
		
		k.predict();
		a.predict();

		float err = ((1000 - (rand() % 2000)) * 1e-3);
		float p = pos_sp + err * 0.002f;
		k.input_position(p);

		Matrix<float, 3, 1> xk = k.get_prediction(); 

		float pos = xk(0);
		float vel = xk(1) * 1000;
		float acc = xk(2) * 1000000;

		//k.input_position(pos_sp);

		a.input_position(vel);
		Matrix<float, 3, 1> xka = a.get_prediction(); 

		//acc = xka(1) * 1000;
		if(fabsf(vel) > 2) vel = 0;
		if(fabsf(pos) > 4) pos = 0;
		printf("time = [time; %f];\n", (float)it);
		printf("vel_sp = [vel_sp; %f];\n", vel_sp);
		printf("pos_sp = [pos_sp; %f];\n", pos_sp);
		printf("pos_x = [pos_x; %f];\n", pos);
		printf("vel_x = [vel_x; %f];\n", vel);
		printf("acc_x = [acc_x; %f];\n", acc);
		printf("int_pos = [int_pos; %f];\n", int_pos);
	
		int_pos += xk(1);
		// start test after 10 iterations
		//if(it > 10) TEST(is_equal(truth, xk(0), 0.1f)); 
		it++; 
	}

	printf("plot(time, pos_x, time, vel_x, time, pos_sp, time, vel_sp, time, acc_x);\n");
	//printf("plot(time, pos_x, time, vel_x);\n");
	printf("input(\"press any key\");\n");


	//sem_wait(&_shutdown);
	return 0; 
}
