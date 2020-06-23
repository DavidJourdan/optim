// SolverBase.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/16/20

#include "SolverBase.h"

#include "filter_var.h"

#include <iostream> // printf, std::cout, std::cerr

using namespace optim;

template <typename scalar>

void SolverBase<scalar>::main_loop()
{
  while(gradient_norm() > options.threshold && info() == SolverStatus::success)
  {
    solve_one_step();
    if(iteration_count() > options.iteration_limit)
      _status = SolverStatus::iteration_overflow;
  }

  if(info() != SolverStatus::success)
    display_status();
}

template <typename scalar>
void SolverBase<scalar>::init_base(scalar energyVal)
{
  _energyVal = energyVal;
  if(options.display != SolverDisplay::quiet)
    display_header();

  std::sort(options.fixed_dofs.begin(), options.fixed_dofs.end());

  if(std::isnan(_energyVal) || std::isnan(gradient_norm()))
    _status = SolverStatus::NaN_error;
  else
    _status = SolverStatus::success;

  if(options.threshold < 0)
    options.threshold = 1e-4 * gradient_norm();
}

template <typename scalar>
void SolverBase<scalar>::display_header() const
{
  int spacing = 10 + options.display_precision;
  printf("%s\n%s%*s%*s%*s\n", method_name(), "Iter", spacing, "Fval", spacing, "Step size", spacing, "Optimality");
}

template <typename scalar>
void SolverBase<scalar>::display_line(scalar step_size)
{
  int spacing = 10 + options.display_precision;
  printf("%4i%*.*e%*.*e%*.*e\n", iteration_count(), spacing, options.display_precision, _energyVal, spacing,
         options.display_precision, step_size, spacing, options.display_precision, gradient_norm());
}

template <typename scalar>
void SolverBase<scalar>::display_status() const
{
  switch(info())
  {
  case SolverStatus::line_search_failed:
    std::cerr << "Line search failed\n";
    break;
  case SolverStatus::wrong_descent_direction:
    std::cerr << "Error: Not a descent direction\n";
    break;
  case SolverStatus::regularization_failed:
    std::cerr << "Regularization failed\n";
    break;
  case SolverStatus::iteration_overflow:
    std::cerr << "Iteration limit reached\n";
    break;
  case SolverStatus::NaN_error:
    std::cerr << "Error: not a number\n";
    break;
  case SolverStatus::uninitialized:
    std::cerr << "Error: solver is not initialized\n";
    break;
  case SolverStatus::success:
    std::cout << "Solver OK\n";
    break;
  default:
    std::cerr << "Unknown error type\n";
  }
}

template <typename scalar>
scalar SolverBase<scalar>::line_search(Eigen::Ref<const Vec<scalar>> direction)
{
  scalar step_size = 1.0;
  scalar k = options.line_search.armijo_c * gradient_value().dot(direction);
  scalar current_energy_value = energy(var() + step_size * direction);

  while(current_energy_value > _energyVal + step_size * k)
  {
    step_size *= options.line_search.shrink_factor;
    if(step_size < 1e-8)
      return -1;

    current_energy_value = energy(var() + step_size * direction);
  }
  _energyVal = current_energy_value;
  if(std::isnan(_energyVal))
    set_status(SolverStatus::NaN_error);

  return step_size;
}

namespace optim
{
template class SolverBase<double>;
template class SolverBase<float>;
} // namespace optim