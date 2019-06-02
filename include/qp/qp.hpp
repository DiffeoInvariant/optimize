#ifndef QP_HPP
#define QP_HPP

#include <Eigen/Dense>
#include <utility>
#include <boost/mpl/vector.hpp>
#include <type_traits>
#include <optional>
#include <string>
#include <cstdio>


namespace optimize
{
	namespace qp
	{
		enum class qp_trait
		{
			Convex,
			NonConvex,
			Infeasible,
			Feasible,
			Optimal,
			PrimalUnbounded,
			DualUnbounded
		};


		template<typename T>
		using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

		template<typename T>
		using Vec = Eigen::Matrix<T,Eigen::Dynamic, 1>;

		namespace mpl = boost::mpl;



		template<typename T>
		extern Vec<T> SequentialConvexQuadOpt(const Mat<T>& P, const Vec<T>& c, const Mat<T>& G,
									   Vec<T>& s, const Vec<T>& h, const Mat<T>& A,
									   const Vec<T>& b, Vec<T>& xstar);

		#define C_OPT_ARG(...) if(__VA_ARGS__##_) __VA_ARGS__ = *__VA_ARGS__##_


		template<typename T, qp_trait solver_t= qp_trait::NonConvex, typename Enable = std::void_t<>,
				 typename... SolverArgs>
		class QuadraticSolver
		{
		protected:
			/**
			 * variables for solving the QP:
			 *
			 * minimize 0.5 * x^T P x + c^T x
			 * s.t.		Gx + s = h,
			 *			AX = b,
			 *			s >= 0
			 */

			size_t		problem_size = 1;

			Mat<T>		P = Mat<T>::Zero(problem_size, problem_size);

			Vec<T>		c = Vec<T>::Zero(problem_size);
			
			Mat<T>		G = Mat<T>::Zero(problem_size, problem_size);

			Vec<T>		s = Vec<T>::Zero(problem_size);

			Vec<T>		h = Vec<T>::Zero(problem_size);

			Mat<T>		A = Mat<T>::Zero(problem_size, problem_size);

			Mat<T>		b = Vec<T>::Zero(problem_size);

			Vec<T>		xstar = Vec<T>::Zero(problem_size);


		public:

			typedef decltype(solver_t) geometry;

			mpl::vector<SolverArgs...> state;

			template<typename First, typename... Rest>
			void setSolverArgs(typename std::enable_if<std::is_enum<First>::value, First>::type &&first,
								Rest&&... rest);


			constexpr QuadraticSolver() {};

			constexpr QuadraticSolver(size_t probSize);

			QuadraticSolver(QuadraticSolver& other) = default;

			QuadraticSolver(QuadraticSolver&& other) = default;

			/**
			 * params in order: P, c, G, s, h, A, b, xstar, problem_size
			 */
			QuadraticSolver(const Mat<T>& P_,
							const std::optional<Vec<T>>& c_=std::nullopt,
							const std::optional<Mat<T>>& G_=std::nullopt,
							const std::optional<Vec<T>>& s_=std::nullopt,
							const std::optional<Vec<T>>& h_=std::nullopt, 
							const std::optional<Mat<T>>& A_=std::nullopt,
							const std::optional<Vec<T>>& b_=std::nullopt, 
							const std::optional<Vec<T>>& xstar_=std::nullopt,
							const std::optional<size_t>& problem_size_=std::nullopt) : P(P_)
						{

							C_OPT_ARG(problem_size);
							C_OPT_ARG(c);
							C_OPT_ARG(G);
							C_OPT_ARG(s);
							C_OPT_ARG(h);
							C_OPT_ARG(A);
							C_OPT_ARG(b);
							C_OPT_ARG(xstar);
						};


			void setAllSolverObjects(const std::optional<Mat<T>>& P_=std::nullopt,
								const std::optional<Vec<T>>& c_=std::nullopt,
								const std::optional<Mat<T>>& G_=std::nullopt,
								const std::optional<Vec<T>>& s_=std::nullopt,
								const std::optional<Vec<T>>& h_=std::nullopt, 
								const std::optional<Mat<T>>& A_=std::nullopt,
								const std::optional<Vec<T>>& b_=std::nullopt, 
								const std::optional<Vec<T>>& xstar_=std::nullopt,
								const std::optional<size_t>& probSize=std::nullopt)
			{
				if(P_){
					*this = std::move(QuadraticSolver(*P_, c_, G_, s_, h_, A_, b_, xstar_, probSize));
				} else {
					*this = std::move(QuadraticSolver(P, c_, G_, s_, h_, A_, b_, xstar_, probSize));

				}
			}

			
			Vec<T> getXstar(){
				return xstar;
			}

			//returns optimal vector xstar
			Vec<T> optimize(std::string mode="seq")
			{
				Eigen::ColPivHouseholderQR<Mat<T>> AQR(A);
				auto rkA = AQR.rank();
				//PRECONDITION: rank(A) = num_rows(A)
				if(rkA != A.rows()){
					printf("Error: rank of A must equal the number of rows, but rank(A) = %.i.",rkA);
					printf("and nrow(A) = %.i.",A.rows());
				}


				//choose solution mode and solve
				if constexpr(solver_t == qp_trait::Convex){
					if( mode.compare("seq") == 0) {
						return SequentialConvexQuadOpt(P, c ,G, s, h, A, b, xstar);
					} else {
						printf("Error: mode %s not implemented yet. Returning current value of xstar.", mode);
						return xstar;
					}
				} else {
					printf("Error: non-convex problem solutions are not implemented yet. Returning xstar.");
					return xstar;
				}
			}


		};//class QuadraticSolver





	}//namespace qp
}//namespace optimize



#endif//QP_HPP
