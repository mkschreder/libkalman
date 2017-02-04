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
class ConstantVelocityPositionFilter : public IUKFModel<2, 1>{
public: 
	ConstantVelocityPositionFilter(Type _q, Type _r, Type _dt = 1.0f) : _k(this){
		// set initial state
		_k.set_xk(Matrix<Type, 2, 1>((const Type[]){0.0, 0.0})); 

		// setup initial belief
		Matrix<Type, 2, 2> P; 
		Matrix<Type, 2, 2> Q; 
		Matrix<Type, 1, 1> R; 

		// this tells us how much we trust each reading. 
		// lower value means more trust
		// higher value means less trust
		P << 
			1, 0, 
			0, 1; 
		Q = _Q_discrete_white_noise_2(0.001, 0.01);  
		/*Q << 
			1, 0.01,
			0.01, 1;*/
		R << 1.0f; //::pow(_r, 2); 

		// setup initial covariance 
		_k.set_P(P); 

		// setup measurement covariance 
		_k.set_R(R); 

		// setup process noise
		_k.set_Q(Q); 

		this->_dt = _dt;
	}

	void predict(){
		_k.predict(Matrix<Type, 1, 1>()); 
	}

	void input_position(Type val){
		Matrix<Type, 1, 1> m; 
		m << val; 
		_k.update(m); 
	}
	
	void set_prediction(const Matrix<Type, 2, 1> &m){
		_k.set_xk(m); 
	}

	Matrix<Type, 2, 1> get_prediction(){
		return _k.get_prediction();
	}
protected: 
	virtual Matrix<Type, 2, 1> F(const Matrix<Type, 2, 1> &i) override {
		Matrix<float, 2, 1> data; 
		data << 
			i(0) + i(1),
			i(1);
		return data; 
	}
	virtual Matrix<Type, 1, 1> H(const Matrix<Type, 2, 1> &i) override {
		Matrix <float, 1, 1> data;
		data << i(0);
		return data;
	}
private: 
	UnscentedKalmanFilter<2, 1, 1> _k;  

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
