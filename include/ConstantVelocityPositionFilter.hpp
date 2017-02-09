/*
	Copyright (c) 2016 Martin Schröder <mkschreder.uk@gmail.com>

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <math.h>
#include "KalmanFilter.hpp"

namespace Eigen {
namespace filter {

template<typename Type>
class ConstantVelocityPositionFilter : public IUKFModel<3, 1>{
public: 
	ConstantVelocityPositionFilter(Type _q, Type _r, Type _dt = 1.0f) : _k(this){
		// set initial state
		_k.set_xk(Matrix<Type, 3, 1>((const Type[]){0.0, 0.0, 0.0})); 

		// setup initial belief
		Matrix<Type, 3, 3> P; 
		Matrix<Type, 3, 3> Q; 
		Matrix<Type, 1, 1> R; 

		// this tells us how much we trust each reading. 
		// lower value means more trust
		// higher value means less trust
		P << 
			0, 0, 0,
			0, 0, 0,
			0, 0, 0;
		//Q = _Q_discrete_white_noise_2(0.01, 0.001);  
		Q << 
			1e-3, 0.0, 0.0,
			0.0, 1e-4, 0.0,
			0.0, 0.0, 1e-7;

		R << 0.7f; //::pow(_r, 2); 

		// setup initial covariance 
		_k.set_P(P); 

		// setup measurement covariance 
		_k.set_R(R); 

		// setup process noise
		_k.set_Q(Q); 

		this->_dt = _dt;
	}

	void set_R(float r){
		Matrix<float, 1, 1> R;
		R << r;
		_k.set_R(R);
	}

	void set_Q(Matrix<float, 3, 3> &m){
		_k.set_Q(m);
	}

	void predict(){
		_k.predict(Matrix<Type, 1, 1>()); 
	}

	void input_position(Type val){
		Matrix<Type, 1, 1> m;
		m << val;
		_k.update(m);
	}

	void set_prediction(const Matrix<Type, 3, 1> &m){
		_k.set_xk(m); 
	}

	Matrix<Type, 3, 1> get_prediction(){
		return _k.get_prediction();
	}
protected: 
	virtual Matrix<Type, 3, 1> F(const Matrix<Type, 3, 1> &i) override {
		Matrix<float, 3, 1> data;
		data <<
			i(0) + i(1) + i(2),
			i(1) + i(2),
			i(2);
		return data;
	}
	virtual Matrix<Type, 1, 1> H(const Matrix<Type, 3, 1> &i) override {
		Matrix <float, 1, 1> data;
		data << i(0);
		return data;
	}
private: 
	UnscentedKalmanFilter<3, 1, 1> _k;  

	Matrix<Type, 2, 2> _Q_discrete_white_noise_2(Type dt, Type var){
		Matrix<Type, 2, 2> q; 
		q << 
			0.25f*::pow(dt, 4), 0.5 * ::pow(dt, 3), 
			0.50f*::pow(dt, 3), ::pow(dt, 2); 
		return q * var;  
	}

	float _dt;
}; 

}
}
