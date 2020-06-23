// test_newton.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/08/20

#include "catch.hpp"
#include "helpers.h"
#include "optim/LBFGS.h"

TEST_CASE("Rosenbrock function (double)", "[Newton]")
{
  using namespace Eigen;
  using namespace optim;

  const auto energy = [](const auto &x) {
    int n = x.size();
    double res = 0;
    for(int i = 0; i < n; i += 2)
    {
      res += pow(1.0 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
    }
    return res;
  };
  const auto gradient = [](const auto &x) {
    int n = x.size();
    VectorXd grad = VectorXd::Zero(n);
    for(int i = 0; i < n; i += 2)
    {
      grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
      grad(i) = -2 * (x(i) * grad(i + 1) + 1.0 - x(i));
    }
    return grad;
  };

  SECTION("Solve")
  {
    LBFGSSolver<> solver;

    REQUIRE(solver.info() == SolverStatus::uninitialized);

    solver.options.display = SolverDisplay::quiet;
    solver.options.threshold = 1e-4;
    VectorXd x = VectorXd::Constant(100, 3);
    solver.solve(energy, gradient, x);

    REQUIRE_THAT(solver.var(), ApproxEquals<double>(VectorXd::Ones(100)));
  }

  SECTION("Solve 1 step")
  {
    LBFGSSolver<> solver;

    solver.options.display = SolverDisplay::quiet;
    VectorXd x = VectorXd::Constant(100, 3);
    solver.init(energy, gradient, x);

    while(solver.gradient_norm() > 1e-4 && solver.info() == SolverStatus::success)
    {
      solver.solve_one_step();
    }

    REQUIRE_THAT(solver.var(), ApproxEquals<double>(VectorXd::Ones(100)));
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
    };

    LBFGSSolver<> solver;

    solver.options.display = SolverDisplay::quiet;
    solver.options.threshold = 1e-4;
    VectorXd x = VectorXd::Constant(100, 3);

    Rosenbrock r(100);
    solver.solve(r, x);

    REQUIRE_THAT(solver.var(), ApproxEquals<double>(VectorXd::Ones(100)));
  }
}

TEST_CASE("Rosenbrock function (float)", "[Newton]")
{
  using namespace Eigen;
  using namespace optim;

  const auto energy = [](const auto &x) {
    int n = x.size();
    float res = 0;
    for(int i = 0; i < n; i += 2)
    {
      res += pow(1.0 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
    }
    return res;
  };
  const auto gradient = [](const auto &x) {
    int n = x.size();
    VectorXf grad = VectorXf::Zero(n);
    for(int i = 0; i < n; i += 2)
    {
      grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
      grad(i) = -2 * (x(i) * grad(i + 1) + 1.0 - x(i));
    }
    return grad;
  };

  SECTION("Solve")
  {
    LBFGSSolver<float> solver;

    REQUIRE(solver.info() == SolverStatus::uninitialized);

    solver.options.display = SolverDisplay::quiet;
    solver.options.threshold = 1e-4;
    VectorXf x = VectorXf::Constant(100, 3);
    x = solver.solve(energy, gradient, x);

    REQUIRE_THAT(x, ApproxEquals<float>(VectorXf::Ones(100)));
  }

  SECTION("Solve 1 step")
  {
    LBFGSSolver<float> solver;

    solver.options.display = SolverDisplay::quiet;
    VectorXf x = VectorXf::Constant(100, 3);
    solver.init(energy, gradient, x);

    while(solver.gradient_norm() > 1e-4 && solver.info() == SolverStatus::success)
    {
      solver.solve_one_step();
    }

    REQUIRE_THAT(solver.var(), ApproxEquals<float>(VectorXf::Ones(100)));
  }

  SECTION("Object-oriented")
  {
    struct Rosenbrock
    {
      int n;
      Rosenbrock(int n_) : n(n_) {}
      float energy(Ref<const VectorXf> x)
      {
        float res = 0;
        for(int i = 0; i < n; i += 2)
        {
          res += pow(1.0 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
        }
        return res;
      }
      VectorXf gradient(Ref<const VectorXf> x)
      {
        VectorXf grad = VectorXf::Zero(n);
        for(int i = 0; i < n; i += 2)
        {
          grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
          grad(i) = -2 * (x(i) * grad(i + 1) + 1.0 - x(i));
        }
        return grad;
      }
    };

    LBFGSSolver<float> solver;

    solver.options.display = SolverDisplay::quiet;
    solver.options.threshold = 1e-4;
    VectorXf x = VectorXf::Constant(100, 3);

    Rosenbrock r(100);
    solver.solve(r, x);

    REQUIRE_THAT(solver.var(), ApproxEquals<float>(VectorXf::Ones(100)));
  }
}
