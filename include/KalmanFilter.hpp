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

#include <matrix/matrix/Matrix.hpp>
#include <matrix/matrix/SquareMatrix.hpp>
#include <matrix/matrix/Vector.hpp>

// KalmanFilter template (xc = size of state, zc = size of measurement, uc = size of control vector)
template<unsigned int XC, unsigned int ZC, unsigned int UC>
class KalmanFilter {
public: 
	typedef matrix::Matrix<float, XC, XC> StateMatrixType; 	
	typedef matrix::Matrix<float, ZC, ZC> InputMatrixType; 	
	typedef matrix::Matrix<float, ZC, XC> StateInputMatrixType; 	
	typedef matrix::Matrix<float, XC, ZC> GainMatrixType; 	
	typedef matrix::Matrix<float, XC, UC> ControlMatrixType; 	
	typedef matrix::Vector<float, XC> StateVectorType; 
	typedef matrix::Vector<float, ZC> InputVectorType; 
	typedef matrix::Vector<float, UC> ControlVectorType; 

	KalmanFilter(){
		F.setIdentity(); 
		B.setZero(); 
		H.setZero(); 
		P.setIdentity(); 
		R.setIdentity(); 
		Q.setIdentity(); 
		xk.setZero(); 
	}
	void predict(const ControlVectorType &uk){
		// predict
		xk = F * xk + B * uk; 
		P = F * P * F.transposed() + Q; 
	}
	void update(const InputVectorType &zk){
		// observe
		InputVectorType y = zk - H * xk; 
		InputMatrixType S = H * P * H.transposed() + R; 

		// update
		GainMatrixType K = P * H.transposed() * matrix::inversed(S); 
		xk = xk + K * y; 
		P = P - K * H * P; 
	}
	void set_xk(const StateVectorType &vec){
		xk = vec; 
	}
	// process covariance matrix
	void set_P(const StateMatrixType &mat){
		P = mat; 
	}
	// state control function
	void set_B(const StateMatrixType &mat){
		B = mat; 
	}
	// state transition equations matrix
	void set_F(const StateMatrixType &mat){
		F = mat; 
	}
	// measurement noise matrix
	void set_R(const InputMatrixType &mat){
		R = mat; 
	}
	// process noise matrix
	void set_Q(const StateMatrixType &mat){
		Q = mat; 
	}
	// measurement to state transition matrix
	void set_H(const StateInputMatrixType &mat){
		H = mat; 
	}
	const StateVectorType &get_prediction() const {
		return xk; 
	}
private: 
	// state transition matrix
	StateMatrixType F; 
	// external forces matrix
	ControlMatrixType B;
	// input vector to state matrix 
	StateInputMatrixType H;
	// sensor noise
	InputMatrixType R; 
	// process noise 
	StateMatrixType Q; 

	// prediction error matrix
	StateMatrixType P;  

	// filter state 
	StateVectorType xk; 
}; 

