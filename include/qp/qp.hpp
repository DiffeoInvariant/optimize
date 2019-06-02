#ifndef QP_HPP
#define QP_HPP

#include <Eigen/Dense>
#include <utility>
#include <boost/mpl/vector.hpp>
#include <type_traits>

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

		template<typename T, typename solver_t=qp_trait::NonConvex, typename Enable,
				 typename SolverArgs...>
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

			Mat<T>		P = Eigen::Mat<T>::Zero(problem_size, problem_size);

			Vec<T>		c = Eigen::Vec<T>::Zero(problem_size);
			
			Mat<T>		G = Eigen::Mat<T>::Zero(problem_size, problem_size);

			Vec<T>		s = Eigen::Vec<T>::Zero(problem_size);

			Vec<T>		h = Eigen::Vec<T>::Zero(problem_size);

			Mat<T>		A = Eigen::Mat<T>::Zero(problem_size, problem_size);

			Mat<T>		b = Eigen::Vec<T>::Zero(problem_size);

			Vec<T>		xstar = Eigen::Vec<T>::Zero(problem_size);


		public:

			typedef solver_t geometry;

			mpl::vector<SolverArgs...> state;

			template<typename First, typename Rest...>
			void setSolverArgs(typename std::enable_if<std::is_enum<First>::value, First>::type &&first,
								Rest&&... rest);


			constexpr QuadraticSolver() {};

			constexpr QuadraticSolver(size_t probSize);

			QuadraticSolver(QuadraticSolver other) = default;

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
							const std::optional<size_t>& probSize=std::nullopt);

			Vec<T> getXstar();

			mpl::vector<SolverArgs...> getState();

			//returns optimal vector xstar
			Vec<T> optimize();


			void setSolverObjects(const std::optional<Mat<T>>& P_=std::nullopt,
								const std::optional<Vec<T>>& c_=std::nullopt,
								const std::optional<Mat<T>>& G_=std::nullopt,
								const std::optional<Vec<T>>& s_=std::nullopt,
								const std::optional<Vec<T>>& h_=std::nullopt, 
								const std::optional<Mat<T>>& A_=std::nullopt,
								const std::optional<Vec<T>>& b_=std::nullopt, 
								const std::optional<Vec<T>>& xstar_=std::nullopt,
								const std::optional<size_t>& probSize=std::nullopt);

			
		};




	}//namespace qp
}//namespace optimize



#endif//QP_HPP
