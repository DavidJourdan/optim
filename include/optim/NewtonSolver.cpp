// NewtonSolver.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include "NewtonSolver.h"

#include "filter_var.h"

#include <future>   // future
#include <iostream> // printf, std::cout, std::cerr
#include <limits>   // std::numeric_limits<scalar>::max()

using namespace optim;

template <typename scalar>
NewtonSolver<scalar>::NewtonSolver() : _regularization_coeff{0.01}, _status{success}
{}

template <typename scalar>
Vec<scalar> NewtonSolver<scalar>::solve_one_step()
{
  using namespace Eigen;

  if(_regularization_coeff > options.regularization.min) // start from a smaller value than in the previous iteration
    _regularization_coeff *= options.regularization.shrink_factor;
  else
    _regularization_coeff = options.regularization.min;

  scalar min = min_diagonal(_hessian_val);

  if(min < options.regularization.tolerance)
  {
    _regularization_coeff = -min + options.regularization.beta; // sets the smallest coefficient on the diagonal to beta
    regularize(_hessian_val);
  }

  // solve for _hessian_val*dx = _force_val and regularize if needed
  auto dx = linear_solve(_hessian_val, _force_val);

  if(dx.size() > 0)
  {
    // perform a line search to best minimize the energy
    auto alpha = line_search(dx);

    if(alpha > 0)
    {
      update(alpha, dx);

      if(options.display != quiet)
        display_line(alpha);
    }
    else
      _status = line_search_failed;
  }
  return _var;
}

template <typename scalar>
void NewtonSolver<scalar>::update(scalar alpha, Eigen::Ref<const Vec<scalar>> dx)
{
  _var += alpha * dx;

  options.update_fct(_var);

  if(options.compute_parallel)
  {
    auto hessian_future = std::async(
        std::launch::async, [this](const auto &x) { return hessian(x); }, _var);
    _force_val = -gradient(_var);
    _hessian_val = hessian_future.get();
  }
  else
  {
    _force_val = -gradient(_var);
    _hessian_val = hessian(_var);
  }
  if(std::isnan(_force_val.sum()) || std::isnan(_hessian_val.sum()))
    _status = NaN_error;
}

template <typename scalar>
void NewtonSolver<scalar>::display_header() const
{
  int spacing = 10 + options.display_precision;
  printf("NEWTON METHOD\n%s%*s%*s%*s\n", "Iter", spacing, "Fval", spacing, "Step size", spacing, "Optimality");
}

template <typename scalar>
void NewtonSolver<scalar>::display_line(scalar step_size)
{
  int spacing = 10 + options.display_precision;
  printf("%4i%*.*e%*.*e%*.*e\n", _iter++, spacing, options.display_precision, _energy_val, spacing,
         options.display_precision, step_size, spacing, options.display_precision, gradient_norm());
}

template <typename scalar>
void NewtonSolver<scalar>::display_status() const
{
  switch(info())
  {
  case line_search_failed:
    std::cerr << "Line search failed\n";
    break;
  case wrong_descent_direction:
    std::cerr << "Error: Not a descent direction\n";
    break;
  case regularization_failed:
    std::cerr << "Regularization failed\n";
    break;
  case iteration_overflow:
    std::cerr << "Iteration limit reached\n";
    break;
  case NaN_error:
    std::cerr << "Error: not a number\n";
    break;
  case success:
    std::cout << "Solver OK\n";
  default:
    std::cerr << "Unknown error type\n";
  }
}

template <typename scalar>
void NewtonSolver<scalar>::regularize(Eigen::SparseMatrix<scalar> &mat) const
{
  for(int j = 0; j < mat.cols(); ++j)
  {
    mat.coeffRef(j, j) += _regularization_coeff;
  }
  for(int i: options.fixed_dofs)
    mat.coeffRef(i, i) = 1;
}

template <typename scalar>
scalar NewtonSolver<scalar>::min_diagonal(Eigen::SparseMatrix<scalar> &mat)
{
  scalar min = std::numeric_limits<scalar>::max();
  for(int i = 0; i < mat.cols(); ++i)
  {
    if(mat.coeff(i, i) < min)
      min = mat.coeff(i, i);
  }
  return min;
}

template <typename scalar>
Vec<scalar> NewtonSolver<scalar>::linear_solve(Eigen::SparseMatrix<scalar> &A, Eigen::Ref<const Vec<scalar>> b)
{
  using namespace Eigen;

  _solver.factorize(A);

  Vec<scalar> x;
  while(_solver.info() != Success)
  {
    if(options.display == verbose)
    {
      std::cerr << "WARNING Not sufficiently positive definite hessian: adding a multiple of identity" << std::endl;
      std::cout << "Regularization coefficent: " << _regularization_coeff << std::endl;
    }
    regularize(A);

    _solver.factorize(A);

    _regularization_coeff *= 2;
    if(_regularization_coeff > options.regularization.max)
    {
      _status = regularization_failed;
      return Vec<scalar>{};
    }
  }

  x = _solver.solve(b);
  assert(_solver.info() == Success && "Solving failed");

  // make sure x is a descent direction
  scalar dot_prod = x.dot(b);
  while(dot_prod < 0)
  {
    if(options.display == verbose)
    {
      std::cerr << "Invalid descent direction: <∇W, direction> = " << -dot_prod << std::endl;
      std::cout << "Regularization coefficent: " << _regularization_coeff << std::endl;
    }

    regularize(A);
    _solver.factorize(A);
    x = _solver.solve(b);

    _regularization_coeff *= 2;
    if(_regularization_coeff > options.regularization.max)
    {
      _status = wrong_descent_direction;
      return Vec<scalar>{};
    }

    dot_prod = x.dot(b);
    if(options.display == verbose)
      std::cout << "<∇W, direction> = " << -dot_prod << std::endl;
  }
  return x;
}

template <typename scalar>
scalar NewtonSolver<scalar>::line_search(Eigen::Ref<const Vec<scalar>> direction)
{
  scalar step_size = 1.0;
  scalar k = options.line_search.armijo_c * (-_force_val).dot(direction);
  scalar current_energy_value = energy(_var + step_size * direction);

  while(current_energy_value > _energy_val + step_size * k)
  {
    step_size *= options.line_search.shrink_factor;
    if(step_size < 1e-8)
      return -1;

    current_energy_value = energy(_var + step_size * direction);
  }
  _energy_val = current_energy_value;
  if(std::isnan(_energy_val))
    _status = NaN_error;

  return step_size;
}

template <typename scalar>
scalar NewtonSolver<scalar>::energy(Eigen::Ref<const Vec<scalar>> x)
{
  return _energy_fct(x);
}
template <typename scalar>
Vec<scalar> NewtonSolver<scalar>::gradient(Eigen::Ref<const Vec<scalar>> x)
{
  Vec<scalar> res = _gradient_fct(x);
  filter_var(res, options.fixed_dofs);
  return res;
}
template <typename scalar>
Eigen::SparseMatrix<scalar> NewtonSolver<scalar>::hessian(Eigen::Ref<const Vec<scalar>> x)
{
  Eigen::SparseMatrix<scalar> res = _hessian_fct(x);
  filter_var(res, options.fixed_dofs);
  return res;
}

namespace optim
{
template class NewtonSolver<double>;

#ifndef OPTIM_USE_CHOLMOD
template class NewtonSolver<float>;
#endif
} // namespace optim