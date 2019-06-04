
#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include "../../include/qp/qp.hpp"
#include <cstdio>
#include <cmath>
#include <utility>
#include <algorithm>
#include <functional>
#include <vector>
#include <Eigen/StdVector>
/*
 *for details, see
 http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
 */
namespace optimize
{
	namespace qp
	{

		template<typename T>
		static Mat<T> make_PAG_mat(const Mat<T>& P, const Mat<T>& A, const Mat<T>& G)
		{
			Mat<T> bigmat(P.rows() + A.rows() + G.rows(), P.cols() +
					A.rows() + G.rows());
			//comma initializer
			bigmat << P, A.transpose(),						G.transpose(),
					  A, Mat<T>::Zero(A.rows(), A.rows()),  Mat<T>::Zero(A.rows(), G.rows()),
					  G, Mat<T>::Zero(G.rows(), A.rows()),  Mat<T>::Zero(G.rows(), G.rows());
			return bigmat;
		}

		template<typename T>
		static Vec<T> make_xyz_vec(const Vec<T>& x, const Vec<T>& y, const Vec<T>& z)
		{
			Vec<T> bigvec(3*x.size());
			bigvec << x,
				      y,
					  z;
			return bigvec;
		}

		//cbh is result of make_xyz_vec(c,-b,-h), bigS result of make_xyz_vec(0,0,s)
		template<typename T>
		static Vec<T> resids(const Mat<T>& PAGMat, const Vec<T>& xyz, const Vec<T>& cbh, const Vec<T>& bigS)
		{
			return bigS + PAGMat * xyz + cbh;
		}
		//returns (s-tilde, z-tilde)
		template<typename T>
		static std::pair<Vec<T>, Vec<T>> primal_dual_scale(const Mat<T>& W,
														   const Vec<T>& s, const Vec<T>& z)
		{
			return std::make_pair((W.transpose()).solve(s), W*z);
		}

		template<typename T>
		static Vec<T> vec_of_mat(Mat<T> matrix){
			//multiply diagonal by 1/sqrt(2) = sqrt(2)/2
			matrix.diagonal() *= (0.5 * sqrt(2.0));

			matrix *= sqrt(2.0);

			Eigen::Map<Vec<T>> vec(matrix.data(), matrix.size());

			return vec;
		}

		template<typename T>
		static Mat<T> mat_of_vec(Vec<T> vect){
			int vecLen = vect. template size();
			// solves x*(x+1) = k, which has solution 
			// x = 0.5*[ -1 +/- sqrt(1+4k)]. returns positive solution only
			auto solve_x2p1 = [](T k) -> int {

				T soln = 0.5 * (sqrt(1.0 + 4.0*k) - 1);
				//check if solution is (close enough to) an integer
				return abs(soln - static_cast<int>(soln)) < 1.0e-6 ?
											static_cast<int>(soln) :
											-1;
				};

			int p = solve_x2p1(static_cast<T>(2*vecLen));
			
			if( p == -1){
				printf("Error: length of vector is not a multiple of p(p+1)/2 for some p.");
				return Mat<T>::Zero(vecLen, vecLen);
			}

			Mat<T> matrix(p,p);
			matrix = matrix.template selfAdjointView<Eigen::Upper>();
			double invrt2 = 0.5 * sqrt(2.0);
			for(int j = 0; j < p; j++){
				for(int i = j; i < p; i++){
					if(i == 0 or j == 0){
						auto elem = (i == j) ? vect(0) : invrt2 * vect(std::max(i,j));
						matrix(i,j) = elem;					}
					else if(i == j){
						matrix(i,j) = vect(i*(p+1)/2);
					} else {
						auto elem = invrt2 * vect(i*(p + 1)/2 + p - j);
						matrix(i,j) = elem;
					}
				}//end inner for
			}//end outer for
			return matrix;
		}

		enum class BarrierType
		{
			Log
		};

		enum class Space
		{
			Rpp, //positive vectors in the cone in R^p
			Qp, // magnitude, direction representation of a vector in R(p-1)p (second-order cone in Rp)
			Sp //vec_of_mat(mat) taking pos. semi-definite argument mat

		};

		template<typename T, typename cont>
		static T logBarrierFunc(cont u, Space space){
			auto ln = [] (T x) -> T { return log(x); };
			switch(space)
			{
				case Space::Rpp:
				{
					//cont is a Vec<T>
					T sum = 0;
					for(auto it : u){
						sum += ln(*it);
					}
					return -1*sum;
					break;
				}
				case Space::Qp:
				{
					//cont is a std::pair<Vec<T>, T>, with second element u0, an upper bound on 2-norm
					T u0 = u.second();
					Vec<T> u1 = u.first();
					return -0.5 * log(u0 * u0 - u1.squaredNorm());
					break;
				}
				case Space::Sp:
				{
					//u is a Mat<T>, result of calling mat_of_vec(uvec) 
					//(or a Vec<T>, and we'll call it here)
					//TODO: find a better way to do this with no determinant
					if constexpr(! std::is_same<cont, Mat<T>>::value){
						return -1.0*log(mat_of_vec(u).determinant());
					} else {
						return -1.0*log(u.determinant());
						break;
					}
				}

			}

		}


		template<typename T, typename cont>
		static T barrierFunc(const cont& u, std::string space, BarrierType which){
			switch(which)
			{
				case BarrierType::Log:
					return logBarrierFunc(u, space);
				default:
					printf("Error: not implemented yet.");
					return -1;
			}
		}

		template<typename T>
		static Vec<T> gradient(const Vec<T>& u, Space space, const std::optional<Vec<T>>& one=std::nullopt,
							   const std::optional<Mat<T>>& J=std::nullopt)
		{
			switch(space)
			{
				case Space::Rpp:
				{
					if(not one){
						printf("Error: no ones vector given. Returning u.");
						return u;
					}
					return -1 * (u.diaonal().unaryExpr(
											std::function<T(T)>(
												[](T x){return 1.0/x;})
											)
								) * (*one);
				}
				case Space::Qp:
				{
					if(not J){
						printf("Error: no J matrix given. Returning u.");
						return u;
					}

					T denom = u.transpose() * (*J) * u;
					return (-1.0/denom) * (*J) * u;
				}
				case Space::Sp:
				{
					//TODO: do this without inverse calculation and vec_of_mat construction
					return -1 * vec_of_mat(mat_of_vec(u).inverse());
				}
			}
		}

		template<typename T>
		static Mat<T> make_Jmat(int p){
			Mat<T> J(p,p);

			J.topRows(1) = std::move(Mat<T>::Zero(1,p));

			J.leftCols(1) = std::move(Mat<T>::Zero(p,1));

			J(0,0) = 1;

			J.block(1,1,p-1,p-1).noalias() = -1* Mat<T>::Identity(p-1,p-1);

			return J;
		}

		template<typename T>
		static Mat<T> Hess_k(const Vec<T>& u, Space space, const std::optional<Mat<T>>& J=std::nullopt)
		{
			switch(space)
			{
				case Space::Rpp:
				{
					return u.diagonal().unaryExpr(
								std::function<T(T)>(
									[](T x){ return 1.0 / (x*x); }
									)
								);
				}
				case Space::Qp:
				{
					if(not J){
						printf("Error: no J matrix provided. Returning identity.");
						return Mat<T>::Identity(u.size(), u.size());
					}

					auto uTJ = u.transpose() * J;

					T scale = uTJ * u;

					auto left = 2*J * u * uTJ;
					return (left - scale * J)/(scale * scale);
				}
				case Space::Sp:
				{
					printf("Error: call vecHess_k for space = Sp. returning identity.");
					return Mat<T>::Identity(u.size(), u.size());
				}
			}
		}

		//TODO: implement
		template<typename T>
		static Vec<T> vecHess_k(const Vec<T>& u, const Vec<T>& v);


		template<typename T>
		Vec<T> J_normalize(const Vec<T>& vec, const Mat<T>& J){
			return vec / sqrt(vec.transpose() * J * vec);
		}

		template<typename T>
		T NesterovToddGamma(const Vec<T>& z, const Vec<T>& s, const Mat<T>& J){
			return sqrt( 0.5*(1+J_normalize(z, J).transpose() * J_normalize(s,J)));
		}

		template<typename T>
		Vec<T> compute_dual_var(Vec<T> s, Mat<T> J, T mu){
			auto js = J * s;
			return mu * js / (s.transpose() * js);
		}

		template<typename T>
		Vec<T> oprod(Vec<T> u, Vec<T> v, Space space,
				std::optional<std::pair<T,T>> lens=std::nullopt)
		{
			switch(space)
			{
				case Space::Rpp:
				{
					return u.cwiseProduct(v);
				}
				case Space::Qp:
				{
					if(not lens){
						printf("Error: for space Qp, lens parameter is needed. Returning u.");
						return u;
					}
					
					return lens.first * v + lens.second * u;
				}
				case Space::Sp:
				{
					auto matu = mat_of_vec(u);
					auto matv = mat_of_vec(v);
					return 0.5 * vec_of_mat(matu *  matv + matv * matu);
				}
			}
		}

		template<typename T>
		static T make_e_k(int p, Space space)
		{
			switch(space)
			{
				case Space::Rpp:
				{
					//T is Vec<U>
					return T::Ones(p);
				}
				case Space::Qp:
				{
					//T is Vec<U>
					auto ek = T::Zero(p);
					ek(0) = 1;
					return ek;
				}
				case Space::Sp:
				{
					//T is Mat<U>
					return T::Identity(p,p);
				}
			}
		}

		//T is Vec<U> or Mat<U>. p is a vector of length k of dimensions
		template<typename T>
		static std::vector<T> make_e_vec(int k, const std::vector<int>& p, Space space)
		{
			std::vector<T> e(k);
			int i = 0;
			for(auto& it : e){
				it = make_e_k<T>(p[i], space);
			}
			return e;
		}

		template<typename T>
		static Vec<T> vecSqrtRpp(const Vec<T>& vec)
		{
			return vec.unaryExpr(std::function<T(T)>(sqrt));
		}


		template<typename T>
		static std::pair<T,Vec<T>> vecSqrtQp(const std::pair<T, Vec<T>>& u, const Mat<T>& J)
		{
			T uju = u.second.transpose() * J * u.second;

			T factor = 1.0 / (2 * (u.first + sqrt(uju)));
			return make_pair(factor * (u.first + sqrt(uju)), factor * u.second);
		}

		template<typename T>
		static Vec<T> NesterovToddScalePointRpp(const Vec<T>& s, const Vec<T>& z)
		{
			return oprod(vecSqrtRpp(s), z.unaryExpr(std::function<T(T)>([](T x){return 1.0/sqrt(x);})),
					Space::Rpp);
		}


			
		template<typename T>
		Vec<T> SequentialConvexQuadOpt(const Mat<T>& P, const Vec<T>& c, const Mat<T>& G,
									   Vec<T>& s, const Vec<T>& h, const Mat<T>& A,
									   const Vec<T>& b, Vec<T>& xstar)
		{
		#ifndef NDEBUG
			Eigen::ColPivHouseholderQR<Mat<T>> AQR(A);
			auto rkA = AQR.rank();
			//PRECONDITION: rank(A) = num_rows(A)
			if(rkA != A.rows()){
				printf("Error: rank of A must equal the number of rows, but rank(A) = %.i.",rkA);
				printf("and nrow(A) = %.i.",A.rows());
				return xstar;
			}
						
			auto n = P.cols();
			//PRECONDITION: P is square
			if(n != P.rows()){
				printf("Error: P must be square.");
				return xstar;
			}

			//initialize with width equal to sum of widths of P, A^T, and G^T
			Mat<T> wideConcat(n, P.cols() + (A.transpose()).cols() +
					(G.transpose()).cols());
			wideConcat << P, A.transpose(), G.transpose();//comma initializer

			Eigen::ColPivHouseholderQR<Mat<T>> QRSlvr(wideConcat);
			auto rnk = QRSlvr.rank();

			//PRECONDITION: rank(wideConcat) = n
			if(rnk != n){
				printf("Error: rank(wideConcat) must equal n, but is %i.",rnk);
				return xstar;
			}
		#endif
		}


	}//namespace qp
}//namespace optimize
