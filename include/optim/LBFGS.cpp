// LBFGS.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/15/20

#include "LBFGS.h"

#include "filter_var.h"

#include <iostream>

using namespace optim;

template <typename scalar>
Vec<scalar> LBFGSSolver<scalar>::solve_one_step()
{
  using namespace Eigen;

  if(this->info() == SolverStatus::uninitialized)
  {
    std::cerr << "Error: uninitialized solver\nCall init() before solve_one_step()\n";
    return Vec<scalar>{};
  }

  Vec<scalar> dir = -_grad;

  // variable names follow Alogrithm 7.4 [Nocedal & Wright 2006]
  const int k = _iter - 1;
  const int m = std::min<int>(this->options.bfgs.m, k);

  if(k > 0)
  {
    // L-BFGS two-loop recursion
    Vec<double> alpha(m);
    for(int i = k - 1; i >= k - m; --i)
    {
      alpha(i % this->options.bfgs.m) = s_col(i).dot(dir) / y_dot_s(i);
      dir -= alpha(i % this->options.bfgs.m) * y_col(i);
    }

    dir *= y_dot_s(k - 1) / y_col(k - 1).squaredNorm(); // Equation 7.20

    for(int i = k - m; i <= k - 1; ++i)
    {
      const scalar beta = y_col(i).dot(dir) / y_dot_s(i);
      dir += (alpha(i % this->options.bfgs.m) - beta) * s_col(i);
    }

    if(!(dir.dot(_grad) < 0)) // also checks if descent direction is NaN this way
    {
      if(std::isnan(_grad.sum()))
      {
        this->set_status(SolverStatus::NaN_error);
        return Vec<scalar>{};
      }

      if(this->options.display == SolverDisplay::verbose)
      {
        std::cerr << "WARNING not a descent direction\nFalling back to gradient descent for this iteration\n";
      }
      dir = -_grad;
    }
  }

  // Save the curent var and gradient
  _oldVar = _var;
  _oldGrad = _grad;

  // perform a line search to best minimize the energy
  scalar step = this->line_search(dir);

  if(step > 0)
  {
    _var += step * dir;
    update_s(k); // s_k = var_k - var_{k-1}

    this->options.update_fct(_var);

    _grad = gradient(_var);
    update_y(k); // y_k = grad_k - grad_{k-1}
    compute_y_dot_s(k);

    if(this->options.display != SolverDisplay::quiet)
      this->display_line(step);
    ++_iter;
  }
  else
    this->set_status(SolverStatus::line_search_failed);

  return _var;
}

template <typename scalar>
scalar LBFGSSolver<scalar>::energy(Eigen::Ref<const Vec<scalar>> x)
{
  return _energy_fct(x);
}

template <typename scalar>
Vec<scalar> LBFGSSolver<scalar>::gradient(Eigen::Ref<const Vec<scalar>> x)
{
  Vec<scalar> res = _gradient_fct(x);
  filter_var(res, this->options.fixed_dofs);
  return res;
}

namespace optim
{
template class LBFGSSolver<double>;
template class LBFGSSolver<float>;
} // namespace optim
