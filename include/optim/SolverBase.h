// SolverBase.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/16/20

#pragma once

#include <Eigen/Dense>

#include <vector>

namespace optim
{

template <typename scalar>
using Vec = Eigen::Matrix<scalar, -1, 1>;

enum class SolverStatus {
  success,
  line_search_failed,
  wrong_descent_direction,
  regularization_failed,
  iteration_overflow,
  NaN_error,
  uninitialized
};

enum class SolverDisplay { quiet, normal, verbose };

template <typename scalar = double>
class SolverBase
{
public:
  /**
   * @brief Construct a new Solver object
   * you can either initialize it by calling init and then call solve_one_step() as many time as you want,
   * or call directly solve
   */
  SolverBase() : _status{SolverStatus::uninitialized} {};

  /**
   * @brief  run the main loop: do solve_one_step() until the stopping condition is met or an error is encountered
   */
  void main_loop();

  /**
   * @brief
   * this function needs to be called AFTER setting the various function callbacks (energy and its derivatives if any)
   * and BEFORE EVERYTHING ELSE, so that gradient_norm() returns a meaningful value
   * @param energyVal
   */
  void init_base(scalar energyVal)
  {
    _energyVal = energyVal;
    if(options.display != SolverDisplay::quiet)
      display_header();

    std::sort(options.fixed_dofs.begin(), options.fixed_dofs.end());

    if(std::isnan(_energyVal))
      _status = SolverStatus::NaN_error;
    else
      _status = SolverStatus::success;

    if(options.threshold < 0)
      options.threshold = 1e-4 * gradient_norm();
  }

  /**
   * @brief  do one step of the algorithm
   *
   * @return the result of doing on step of the algorithm (ie x_{k+1} = x_k - \alpha * dx where dx is the computed
   * descent direction)
   */
  virtual Vec<scalar> solve_one_step() = 0;

  virtual const char *method_name() const = 0;

  // getters
  virtual int iteration_count() const = 0;
  virtual scalar gradient_norm() const = 0;
  virtual Vec<scalar> var() const = 0;
  virtual Vec<scalar> gradient_value() const = 0;
  scalar energy_value() const { return _energyVal; }

  SolverStatus info() const { return _status; }
  void set_status(SolverStatus s) { _status = s; }

  virtual scalar energy(Eigen::Ref<const Vec<scalar>> x) = 0;

  struct Options
  {
    std::vector<int> fixed_dofs = {};
    std::function<void(Eigen::Ref<const Vec<scalar>>)> update_fct = [](const auto &) {};

    // only useful if not using solve_one_step
    scalar threshold = -1;

    // number of digits to be displayed in display_line
    int display_precision = 3;

    // max number of iterations
    int iteration_limit = 1000;

    SolverDisplay display = SolverDisplay::verbose;

    struct LineSearchOptions
    {
      // coefficient for the Armijo rule (to be set between 0 and 1)
      scalar armijo_c = 1e-4;
      // between 0 and 1, says how much the step size decreases in each step
      scalar shrink_factor = 0.5;
    } line_search;

    struct NewtonRegularizationOptions
    {
      // constants defined in Nocedal & Wright p.51
      scalar beta = 1e-3;
      scalar tolerance = 1e-4; // defined as $\tau$

      scalar max = 1e4;  // maximum regularization coefficient to be added before throwing an error
      scalar min = 1e-6; // minimum coefficient
      // between 0 and 0.5, says how much the regularization coefficient decreases in each iteration
      scalar shrink_factor = 0.25;
      // set to true to compute gradient and hessian in parallel
      bool compute_parallel = false;
    } newton;

    struct LBFGSOptions
    {
      // number of corrections to approximate the inverse Hessian matrix.
      // Values less than \c 3 are not recommended. Large values will result in excessive computing time.
      int m = 6;
    } bfgs;
  } options;

protected:
  /**
   * Backtracking line search algorithm (see Numerical Optimization by Nocedal & Wright)
   * @param direction the direction of descent
   * @return the step size alpha if it succeeds, or -1 if the line search failed
   */
  scalar line_search(Eigen::Ref<const Vec<scalar>> direction);

  /**
   * These functions display relevant per-iteration information, in a similar fashion to Matlab solvers
   * Iter: iteration count
   * Fval: current energy value
   * Step size: step size used to decrease along the descent direction
   * Optimality: norm of the gradient
   */
  void display_header() const;
  void display_line(scalar step_size); // warning: this function increases the iteration count by 1
  void display_status() const;

  scalar _energyVal;

private:
  SolverStatus _status;
};

} // namespace optim
