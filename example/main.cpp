#include "optim/LBFGS.h"
#include "optim/NewtonSolver.h"

#include <iostream>

int main()
{
  using namespace Eigen;
  using namespace optim;

  const auto energy = [](const auto &x) {
    int n = x.size();
    double res = 0;
    for(int i = 0; i < n; i += 2)
    {
      res += pow(1 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
    }
    return res;
  };
  const auto gradient = [](const auto &x) {
    int n = x.size();
    VectorXd grad = VectorXd::Zero(n);
    for(int i = 0; i < n; i += 2)
    {
      grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
      grad(i) = -2 * (x(i) * grad(i + 1) + 1 - x(i));
    }
    return grad;
  };
  const auto hessian = [](const auto &x) {
    int n = x.size();
    SparseMatrix<double> mat(n, n);

    for(int i = 0; i < n; i += 2)
    {
      mat.insert(i + 1, i + 1) = 200;
      mat.insert(i + 1, i) = -400 * x(i);
      mat.insert(i, i + 1) = -400 * x(i);
      mat.insert(i, i) = -2 * (200 * x(i + 1) - 600 * x(i) * x(i) - 1);
    }
    return mat;
  };

  // uncomment the corresponding lines to test the Newton Solver
  LBFGSSolver<double> solver;
  // NewtonSolver<double> solver;

  solver.options.threshold = 1e-4;
  VectorXd x = VectorXd::Constant(10, 3);

  std::cout << solver.solve(energy, gradient, x).transpose() << "\n";
  // std::cout << solver.solve(energy, gradient, hessian, x).transpose() << "\n";

  return 0;
}