
#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include "../../include/qp.hpp"

namespace optimize
{
	namespace qp
	{
		#define C_OPT_ARG(...) if(__VA_ARGS__##_) __VA_ARGS__ = __VA_ARGS__##_

		template<typename T, typename SolverArgs...>
		QuadraticSolver<T, qp_trait::Convex,
						typename std::enable_if<std::is_floating_point<T>::value, int> = 0
						>::QuadraticSolver(const Mat<T>& P_,
							const std::optional<Vec<T>>& c_,
							const std::optional<Mat<T>>& G_,
							const std::optional<Vec<T>>& s_,
							const std::optional<Vec<T>>& h_, 
							const std::optional<Mat<T>>& A_,
							const std::optional<Vec<T>>& b_, 
							const std::optional<Vec<T>>& xstar_,
							const std::optional<size_t>& problem_size_) : P(P_)
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



	}//namespace qp
}//namespace optimize

