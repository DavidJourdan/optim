// NewtonSolver.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include "SolverBase.h"

#include <Eigen/Sparse>
#ifdef OPTIM_USE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include <functional>

namespace optim
{

template <typename scalar>
#ifdef OPTIM_USE_CHOLMOD
using LinearSolver = Eigen::CholmodDecomposition<Eigen::SparseMatrix<scalar>, Eigen::Upper>;
#else
using LinearSolver = Eigen::SimplicialLLT<Eigen::SparseMatrix<scalar>, Eigen::Upper>;
#endif

template <typename scalar = double>
class NewtonSolver : public SolverBase<scalar>
{
public:
  /**
   * @brief Construct a new Newton Solver object
   * you can either initialize it by calling init and then call solve_one_step() as many time as you want,
   * or call directly solve
   */
  NewtonSolver() : SolverBase<scalar>::SolverBase{} {};

  /**
   * @brief Set the energy, gradient and hessian functions
   *
   * @param objective_func  function to minimize (takes as input an Eigen::VectorXd of size n)
   * @param gradient_func  gradient of the objective function (returns an Eigen::VectorXd of size n)
   * @param hessian_func  hessian of the objective function (returns a square Eigen::SparseMatrix<scalar> of size n)
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   */
  template <class ScalarFunc, class VectorFunc, class MatrixFunc>
  void init(const ScalarFunc &objective_func,
            const VectorFunc &gradient_func,
            const MatrixFunc &hessian_func,
            Eigen::Ref<const Vec<scalar>> var)
  {
    _iter = 1;

    _energy_fct = objective_func;
    _gradient_fct = gradient_func;
    _hessian_fct = hessian_func;
    _var = var;
    _force_val = -gradient(var);
    _hessian_val = hessian(var);

    if(std::isnan(_hessian_val.sum()))
      this->set_status(SolverStatus::NaN_error);

    this->init_base(energy(var));
    this->options.update_fct(var);

    _regularization_coeff = 0.01;

    _solver.analyzePattern(_hessian_val);
  }

  /**
   * @brief Set the energy, gradient and hessian functions using a class implementing these functions
   *
   * @tparam Obj any type with the methods energy, gradient and hessian
   * @param obj  an object implementing the objective function and its first and second derivatives
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   */
  template <class Obj>
  void init(Obj &obj, Eigen::Ref<const Vec<scalar>> var)
  {
    init([&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.energy(x); },
         [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.gradient(x); },
         [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.hessian(x); }, var);
  }

  /**
   * @brief  solve the full nonlinear optimization problem in an automated manner (look at NewtonOptions to see how to
   * fine tune the behavior of this method)
   * The algorithm will stop when the gradient is close enough to 0 ("close enough" is defined by options.threshold)
   * @param objective_func  function to minimize (takes as input an Eigen::VectorXd of size n)
   * @param gradient_func  gradient of the objective function (returns an Eigen::VectorXd of size n)
   * @param hessian_func  hessian of the objective function (returns a square Eigen::SparseMatrix<scalar> of size n)
   * @param _var  initial guess for the algorithm (Eigen::VectorXd of size n)
   * @return the result of the algorithm (ie x = \argmin f such that \nabla f(x) = 0 and \nabla^2 f(x) is SPD)
   */
  template <class ScalarFunc, class VectorFunc, class MatrixFunc>
  Vec<scalar> solve(const ScalarFunc &objective_func,
                    const VectorFunc &gradient_func,
                    const MatrixFunc &hessian_func,
                    Eigen::Ref<const Vec<scalar>> var)
  {
    init(objective_func, gradient_func, hessian_func, var);

    this->main_loop();

    return _var;
  }

  /**
   * @brief  solve the full nonlinear optimization problem in an automated manner, overload for objects implementing the
   * needed functions
   *
   * @tparam Obj any type with the methods energy, gradient and hessian
   * @param obj  an object implementing the objective function and its first and second derivatives
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   * @return the result of the algorithm (ie x = \argmin f such that \nabla f(x) = 0 and \nabla^2 f(x) is SPD)
   */
  template <class Obj>
  Vec<scalar> solve(Obj &obj, Eigen::Ref<const Vec<scalar>> var)
  {
    return solve([&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.energy(x); },
                 [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.gradient(x); },
                 [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.hessian(x); }, var);
  }

  /**
   * @brief  do one step of the algorithm
   *
   * @return the result of doing one step of the algorithm (ie x_{k+1} = x_k - \alpha * dx where dx is the computed
   * descent direction)
   */
  Vec<scalar> solve_one_step() override;

  const char *method_name() const override { return "NEWTON METHOD"; }

  // getters
  int iteration_count() const override { return _iter; }
  scalar gradient_norm() const override { return _force_val.norm(); }
  Vec<scalar> var() const override { return _var; }
  Vec<scalar> gradient_value() const override { return -_force_val; }

protected:
  /**
   * Adds a multiple of the identity to a matrix
   * @param mat a sparse matrix which may or may not be SPD
   */
  void regularize(Eigen::SparseMatrix<scalar> &mat) const;

  // returns the minimum coefficient on the diagonal
  static scalar min_diagonal(Eigen::SparseMatrix<scalar> &mat);

  /**
   * Updates var (with alpha*dx) and gradient and hessian value
   * @param alpha  step size
   * @param dx  descent direction
   */
  void update(scalar alpha, Eigen::Ref<const Vec<scalar>> dx);

  /**
   * solve Ax = b while checking the positive-definiteness of A and adding a multiple of identity if necessary
   * @param A  sparse matrix (hessian or equivalent)
   * @param b  vector (opposite of the gradient)
   * @return x  the descent direction, or an empty vector if the linear solve failed
   */
  Vec<scalar> linear_solve(Eigen::SparseMatrix<scalar> &A, Eigen::Ref<const Vec<scalar>> b);

  scalar energy(Eigen::Ref<const Vec<scalar>> x) override;
  Vec<scalar> gradient(Eigen::Ref<const Vec<scalar>> x);
  Eigen::SparseMatrix<scalar> hessian(Eigen::Ref<const Vec<scalar>> x);

private:
  Vec<scalar> _var;
  int _iter;
  Vec<scalar> _force_val;
  Eigen::SparseMatrix<scalar> _hessian_val;
  scalar _regularization_coeff;

  LinearSolver<scalar> _solver;
  std::function<scalar(Eigen::Ref<const Vec<scalar>>)> _energy_fct;
  std::function<Vec<scalar>(Eigen::Ref<const Vec<scalar>>)> _gradient_fct;
  std::function<Eigen::SparseMatrix<scalar>(Eigen::Ref<const Vec<scalar>>)> _hessian_fct;
};

} // namespace optim
