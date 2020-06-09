// test_newton.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/08/20

#include "catch.hpp"
#include "helpers.h"
#include "optim/NewtonSolver.h"

TEST_CASE("Rosenbrock function", "[Newton]")
{
  using namespace Eigen;
  using namespace optim;

  const auto energy = [](const auto &x) { return pow(1 - x(0), 2) + 100 * pow(x(1) - x(0) * x(0), 2); };
  const auto gradient = [](const auto &x) {
    return Vector2d(2 * (x(0) - 1 - 200 * x(0) * (x(1) - x(0) * x(0))), 200 * (x(1) - x(0) * x(0)));
  };
  const auto hessian = [](const auto &x) {
    SparseMatrix<double> mat(2, 2);
    mat.insert(0, 0) = 2 * (1 - 200 * (x(1) - x(0) * x(0)) + 400 * x(0) * x(0));
    mat.insert(0, 1) = -400 * x(0);
    mat.insert(1, 0) = -400 * x(0);
    mat.insert(1, 1) = 200;
    return mat;
  };

  SECTION("Solve")
  {
    NewtonSolver<double> solver;

    solver.options.display = NewtonSolver<double>::quiet;
    solver.options.threshold = 1e-4;
    Eigen::Vector2d x(3, 3);
    solver.solve(energy, gradient, hessian, x);

    CHECK(solver.var()(0) == Approx(1).epsilon(1e-4));
    CHECK(solver.var()(1) == Approx(1).epsilon(1e-4));
  }

  SECTION("Solve 1 step")
  {
    NewtonSolver<double> solver;

    solver.options.display = NewtonSolver<double>::quiet;
    Eigen::Vector2d x(3, 3);
    solver.init(energy, gradient, hessian, x);

    while(solver.gradient_norm() > 1e-4 && solver.info() == NewtonSolver<double>::success)
    {
      solver.solve_one_step();
    }

    CHECK(solver.var()(0) == Approx(1).epsilon(1e-4));
    CHECK(solver.var()(1) == Approx(1).epsilon(1e-4));
  }

  SECTION("Object-oriented")
  {
    struct Rosenbrock
    {
      int n;
      Rosenbrock(int n_) : n(n_) {}
      double energy(Ref<const VectorXd> x)
      {
        double res = 0;
        for(int i = 0; i < n; i += 2)
        {
          res += pow(1.0 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
        }
        return res;
      }
      VectorXd gradient(Ref<const VectorXd> x)
      {
        VectorXd grad = VectorXd::Zero(n);
        for(int i = 0; i < n; i += 2)
        {
          grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
          grad(i) = -2 * (x(i) * grad(i + 1) + 1.0 - x(i));
        }
        return grad;
      }
      SparseMatrix<double> hessian(Ref<const VectorXd> x)
      {
        SparseMatrix<double> mat(n, n);
        for(int i = 0; i < n; i += 2)
        {
          mat.insert(i + 1, i + 1) = 200;
          mat.insert(i + 1, i) = -400 * x(i);
          mat.insert(i, i + 1) = -400 * x(i);
          mat.insert(i, i) = -2 * (200 * x(i + 1) - 600 * x(i) * x(i) - 1);
        }
        return mat;
      }
    };

    NewtonSolver<double> solver;

    solver.options.display = NewtonSolver<double>::quiet;
    solver.options.threshold = 1e-4;
    VectorXd x = VectorXd::Constant(10, 3);

    Rosenbrock r(10);
    solver.solve(r, x);

    CHECK(solver.var()(0) == Approx(1).epsilon(1e-4));
    CHECK(solver.var()(1) == Approx(1).epsilon(1e-4));
  }
}
