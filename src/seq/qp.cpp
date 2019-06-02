
#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include "../../include/qp/qp.hpp"
#include <cstdio>
#include <cmath>

/*
 *for details, see
 http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
 */
namespace optimize
{
	namespace qp
	{
		enum class BarrierType
		{
			Log
		};

		enum class Space
		{
			Rpp
		};

		template<typename T>
		static T logBarrierFunc(Vec<T> u, Space space){
			auto ln = [] (T x) -> T { return log(x); };
			switch(space)
			{
				case Space::Rpp:
					T sum = 0;
					for(auto it : u){
						sum += ln(*it);
					}
					return -1*sum;
			}

		}


		template<typename T>
		static T barrierFunc(Vec<T> u, std::string space, BarrierType which){
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
		static Mat<T> make_PAG_mat(const Mat<T>& P, const Mat<T>& A, const Mat<T>& G)
		{
			auto zro = Mat<T>::Zero(P.rows(), P.cols());
			Mat<T> bigmat(3*P.rows(), 3*P.cols());
			//comma initializer
			bigmat << P, A.transpose(), G.transpose(),
					  A, zro,			zro,
					  G, zro,			zro;
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
			Mat<T> wideConcat(n, P.cols() + (A.transpose()).cols() + (G.transpose()).cols());
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
