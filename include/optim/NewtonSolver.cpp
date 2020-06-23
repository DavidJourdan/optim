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
Vec<scalar> NewtonSolver<scalar>::solve_one_step()
{
  using namespace Eigen;

  if(this->info() == SolverStatus::uninitialized)
  {
    std::cerr << "Error: uninitialized solver\nCall init() before solve_one_step()\n";
    return Vec<scalar>{};
  }

  if(_regularization_coeff > this->options.newton.min) // start from a smaller value than in the previous iteration
    _regularization_coeff *= this->options.newton.shrink_factor;
  else
    _regularization_coeff = this->options.newton.min;

  scalar min = min_diagonal(_hessian_val);

  if(min < this->options.newton.tolerance)
  {
    _regularization_coeff = -min + this->options.newton.beta; // sets the smallest coefficient on the diagonal to beta
    regularize(_hessian_val);
  }

  // solve for _hessian_val*dx = _force_val and regularize if needed
  auto dx = linear_solve(_hessian_val, _force_val);

  if(dx.size() > 0)
  {
    // perform a line search to best minimize the energy
    auto alpha = this->line_search(dx);

    if(alpha > 0)
    {
      // compute _var + alpha * dx and gradient(_var) and _hessian(_var)
      update(alpha, dx);

      if(this->options.display != SolverDisplay::quiet)
        this->display_line(alpha);
      ++_iter;
    }
    else
      this->set_status(SolverStatus::line_search_failed);
  }
  return _var;
}

template <typename scalar>
void NewtonSolver<scalar>::update(scalar alpha, Eigen::Ref<const Vec<scalar>> dx)
{
  _var += alpha * dx;

  this->options.update_fct(_var);

  if(this->options.newton.compute_parallel)
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
    this->set_status(SolverStatus::NaN_error);
}

template <typename scalar>
void NewtonSolver<scalar>::regularize(Eigen::SparseMatrix<scalar> &mat) const
{
  for(int j = 0; j < mat.cols(); ++j)
  {
    mat.coeffRef(j, j) += _regularization_coeff;
  }
  for(int i: this->options.fixed_dofs)
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
    if(this->options.display == SolverDisplay::verbose)
    {
      std::cerr << "WARNING Not sufficiently positive definite hessian: adding a multiple of identity" << std::endl;
      std::cout << "Regularization coefficent: " << _regularization_coeff << std::endl;
    }
    regularize(A);

    _solver.factorize(A);

    _regularization_coeff *= 2;
    if(_regularization_coeff > this->options.newton.max)
    {
      this->set_status(SolverStatus::regularization_failed);
      return Vec<scalar>{};
    }
  }

  x = _solver.solve(b);
  assert(_solver.info() == Success && "Solving failed");

  // make sure x is a descent direction
  scalar dot_prod = x.dot(b);
  while(dot_prod < 0)
  {
    if(this->options.display == SolverDisplay::verbose)
    {
      std::cerr << "Invalid descent direction: <∇W, direction> = " << -dot_prod << std::endl;
      std::cout << "Regularization coefficent: " << _regularization_coeff << std::endl;
    }

    regularize(A);
    _solver.factorize(A);
    x = _solver.solve(b);

    _regularization_coeff *= 2;
    if(_regularization_coeff > this->options.newton.max)
    {
      this->set_status(SolverStatus::wrong_descent_direction);
      return Vec<scalar>{};
    }

    dot_prod = x.dot(b);
    if(this->options.display == SolverDisplay::verbose)
      std::cout << "<∇W, direction> = " << -dot_prod << std::endl;
  }
  return x;
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
  filter_var(res, this->options.fixed_dofs);
  return res;
}
template <typename scalar>
Eigen::SparseMatrix<scalar> NewtonSolver<scalar>::hessian(Eigen::Ref<const Vec<scalar>> x)
{
  Eigen::SparseMatrix<scalar> res = _hessian_fct(x);
  filter_var(res, this->options.fixed_dofs);
  return res;
}

namespace optim
{
template class NewtonSolver<double>;

#ifndef OPTIM_USE_CHOLMOD
template class NewtonSolver<float>;
#endif
} // namespace optim
