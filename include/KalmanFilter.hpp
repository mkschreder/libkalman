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

	const StateMatrixType &get_P() const {
		return P; 
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

template<unsigned int x, unsigned int y>
void print_matrix(const char *name, const matrix::Matrix<float, x, y> &m){
	printf("%s = [\n", name); 
	for(size_t i = 0; i < x; i++){
		for(size_t j = 0; j < y; j++){
			printf("%f ", m(i, j)); 
		}
		printf("\n"); 
	}
	printf("]\n"); 
}

template<unsigned int XC, unsigned int ZC, unsigned int UC>
class UnscentedKalmanFilter {
public: 
	#define NUM_SIGMA_POINTS (XC * 2u + 1u)

	typedef matrix::Matrix<float, XC, XC> StateMatrixType; 	
	typedef matrix::Matrix<float, ZC, ZC> InputMatrixType; 	
	typedef matrix::Matrix<float, ZC, XC> StateInputMatrixType; 	
	typedef matrix::Matrix<float, XC, ZC> GainMatrixType; 	
	typedef matrix::Matrix<float, XC, UC> ControlMatrixType; 	
	typedef matrix::Vector<float, XC> StateVectorType; 
	typedef matrix::Vector<float, ZC> InputVectorType; 
	typedef matrix::Vector<float, UC> ControlVectorType; 
	typedef matrix::Vector<float, NUM_SIGMA_POINTS> WeightsVectorType; 
	typedef matrix::Matrix<float, NUM_SIGMA_POINTS, XC> SigmaStateMatrixType; 
	typedef matrix::Matrix<float, NUM_SIGMA_POINTS, ZC> SigmaInputMatrixType; 

	typedef matrix::Vector<float, XC> (*f_proc)(const matrix::Vector<float, XC> &); 
	typedef matrix::Vector<float, ZC> (*h_proc)(const matrix::Vector<float, XC> &); 

	UnscentedKalmanFilter(f_proc f, h_proc h){
		B.setZero(); 
		P.setIdentity(); 
		R.setIdentity(); 
		Q.setIdentity(); 
		xk.setZero(); 
		
		_prediction_fn = f; 
		_measurement_fn = h; 

		// sigma point selection constants
		_alpha = 0.1f; 
		_beta = 2.0f; 
		_kappa = 1.0f; 
		_lambda = _alpha * _alpha * (XC + _kappa) - XC; 
			
		computeWeights(); 
	}

	void computeWeights(){
		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			Wc(i) = Wm(i) = 1.0f / (2.0f * (XC + _lambda)); 
		}
		Wc(0) = _lambda / (XC + _lambda) + (1.0f - _alpha * _alpha + _beta); 
		Wm(0) = _lambda / (XC + _lambda); 
		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			//printf("W[%f, %f]\n", Wc(i), Wm(i)); 
		}
	}

	void predict(const ControlVectorType &uk){
		// predict
		//xk = F * xk + B * uk; 
		//P = F * P * F.transposed() + Q; 

		StateMatrixType U; 
		if(!ldlt((XC + _lambda) * P, U)){
			printf("Cholesky failed!\n"); 
			//exit(1); 
		} else {
			//printf("Cholesky success!\n"); 
		}

		SigmaStateMatrixType sigmas; 
		sigmas.setRow(0, xk.transposed()); 
		for(size_t k = 0; k < XC; k++){
			sigmas.setRow(k + 1, xk.transposed() + U.getRow(k)); 
			sigmas.setRow(XC + k + 1, xk.transposed() - U.getRow(k)); 
		}
		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			//printf("E[%d, %f, %f]\n", i, sigmas(i, 0), sigmas(i, 1)); 
		}

		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			_sigmas_f.setRow(i, _prediction_fn(sigmas.getRow(i).transposed()).transposed()); 
		}

		for(size_t i = 0; i < XC; i++){
			xp(i) = Wm.dot(_sigmas_f.getCol(i)); 
		}

		Pp.setZero(); 
		for(size_t k = 0;  k < NUM_SIGMA_POINTS; k++){
			StateVectorType y = _sigmas_f.getRow(k).transposed() - xp; 
			Pp += Wc(k) * (y * y.transposed()); 
		}
		Pp += Q; 
	}

	void update(const InputVectorType &zk){
		// observe
		//InputVectorType y = zk - H * xk; 
		//InputMatrixType S = H * P * H.transposed() + R; 

		// update
		//GainMatrixType K = P * H.transposed() * matrix::inversed(S); 
		//xk = xk + K * y; 
		//P = P - K * H * P; 
		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			_sigmas_h.setRow(i, _measurement_fn(_sigmas_f.getRow(i).transposed())); 
		}

		// transform measurements
		InputVectorType zp; 

		for(size_t i = 0; i < ZC; i++){
			zp(i) = Wm.dot(_sigmas_h.getCol(i)); 
		}

		InputMatrixType Pz; 
		Pz.setZero(); 
		for(size_t k = 0;  k < NUM_SIGMA_POINTS; k++){
			InputVectorType y = _sigmas_h.getRow(k) - zp.transposed(); 
			Pz += Wc(k) * (y * y.transposed()); 
		}
		Pz += R; 
		
		GainMatrixType Pxz; 
		Pxz.setZero(); 
		for(size_t i = 0; i < NUM_SIGMA_POINTS; i++){
			Pxz += Wc(i) * (_sigmas_f.getRow(i).transposed() - xp) * ((_sigmas_h.getRow(i).transposed() - zp)); 
		}

		GainMatrixType K = Pxz * inversed(Pz); 
		xk = xp + K * (zk - zp); 
		P = Pp - K * Pz * K.transposed(); 

		//StateMatrixType U; 
		//print_matrix<2, 2>("P", P); 
		//if(!ldlt(P, U)){ printf("Cholesky failed!\n"); exit(1); }
		//print_matrix<2, 2>("U", U); 
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
	// measurement noise matrix
	void set_R(const InputMatrixType &mat){
		R = mat; 
	}
	// process noise matrix
	void set_Q(const StateMatrixType &mat){
		Q = mat; 
	}

	const StateMatrixType &get_P() const {
		return P; 
	}

	const StateVectorType &get_prediction() const {
		return xk; 
	}
private: 
	// external forces matrix
	ControlMatrixType B;
	// sensor noise
	InputMatrixType R; 
	// process noise 
	StateMatrixType Q; 

	// prediction error matrix
	StateMatrixType P;  

	// filter state 
	StateVectorType xk, xp; 

	StateMatrixType Pp; 

	// Weights
	WeightsVectorType Wm; 
	WeightsVectorType Wc; 

	// Sigma points
	SigmaStateMatrixType _sigmas_f; 
	SigmaInputMatrixType _sigmas_h; 

	float _alpha, _beta, _kappa, _lambda; 

	f_proc _prediction_fn; 
	h_proc _measurement_fn; 
}; 

namespace kalman {
namespace ukf {

}
}
