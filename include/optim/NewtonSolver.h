// NewtonSolver.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>
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
class NewtonSolver
{
public:
  /**
   * @brief Construct a new Newton Solver object
   * you can either initialize it by calling init and then call solve_one_step() as many time as you want,
   * or call directly solve
   */
  NewtonSolver() : _status{uninitialized} {};

  /**
   * @brief Set the energy, gradient and hessian functions
   *
   * @param objective_func  function to minimize (takes as input an Eigen::VectorXd of size n)
   * @param gradient_func  gradient of the objective function (returns an Eigen::VectorXd of size n)
   * @param hessian_func  hessian of the objective function (returns a square Eigen::SparseMatrix<double> of size n)
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   */
  template <class ScalarFunc, class VectorFunc, class MatrixFunc>
  void init(const ScalarFunc &objective_func,
            const VectorFunc &gradient_func,
            const MatrixFunc &hessian_func,
            Eigen::Ref<const Vec<scalar>> var)
  {
    _energy_fct = objective_func;
    _gradient_fct = gradient_func;
    _hessian_fct = hessian_func;

    _iter = 1;
    _regularization_coeff = 0.01;

    if(options.display != quiet)
      display_header();

    std::sort(options.fixed_dofs.begin(), options.fixed_dofs.end());

    _var = var;
    _energy_val = energy(var);
    options.update_fct(_var);
    _force_val = -gradient(var);
    _hessian_val = hessian(var);

    if(std::isnan(_energy_val) || std::isnan(_force_val.sum()) || std::isnan(_hessian_val.sum()))
      _status = NaN_error;
    else
      _status = success;

    _solver.analyzePattern(_hessian_val);

    if(options.threshold < 0)
      options.threshold = 1e-4 * gradient_norm();
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
   * @param hessian_func  hessian of the objective function (returns a square Eigen::SparseMatrix<double> of size n)
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

    while(gradient_norm() > options.threshold && info() == success)
    {
      solve_one_step();
      if(_iter > options.iteration_limit)
        _status = iteration_overflow;
    }

    if(info() != success)
      display_status();

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
  Vec<scalar> solve_one_step();

  // getters
  int iteration_count() const { return _iter; }
  scalar gradient_norm() const { return _force_val.norm(); }
  Vec<scalar> var() const { return _var; }

  enum DisplayMode { quiet, normal, verbose };
  enum StatusType {
    success,
    line_search_failed,
    wrong_descent_direction,
    regularization_failed,
    iteration_overflow,
    NaN_error,
    uninitialized
  };
  StatusType info() const { return _status; };

  struct NewtonOptions
  {
    std::vector<int> fixed_dofs = {};
    std::function<void(Eigen::Ref<const Vec<scalar>>)> update_fct = [](const auto &) {};

    struct RegularizationOptions
    {
      // constants defined in Nocedal & Wright p.51
      scalar beta = 1e-3;
      scalar tolerance = 1e-4; // defined as $\tau$

      scalar max = 1e4;  // maximum regularization coefficient to be added before throwing an error
      scalar min = 1e-6; // minimum coefficient
      // between 0 and 0.5, says how much the regularization coefficient decreases in each iteration
      scalar shrink_factor = 0.25;
    } regularization;

    struct LineSearchOptions
    {
      // coefficient for the Armijo rule (to be set between 0 and 1)
      scalar armijo_c = 1e-4;
      // between 0 and 1, says how much the step size decreases in each step
      scalar shrink_factor = 0.5;
    } line_search;

    // only useful if not using solve_one_step
    scalar threshold = -1;

    // set to true to compute gradient and hessian in parallel
    bool compute_parallel = false;

    // number of digits to be displayed in display_line
    int display_precision = 3;

    // max number of iterations
    int iteration_limit = 5000;

    DisplayMode display = verbose;
  } options;

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
   * Backtracking line search algorithm (see Numerical Optimization by Nocedal & Wright)
   * @param direction the direction of descent
   * @return the step size alpha if it succeeds, or -1 if the line search failed
   */
  scalar line_search(Eigen::Ref<const Vec<scalar>> direction);

  /**
   * solve Ax = b while checking the positive-definiteness of A and adding a multiple of identity if necessary
   * @param A  sparse matrix (hessian or equivalent)
   * @param b  vector (opposite of the gradient)
   * @return x  the descent direction, or an empty vector if the linear solve failed
   */
  Vec<scalar> linear_solve(Eigen::SparseMatrix<scalar> &A, Eigen::Ref<const Vec<scalar>> b);

  /**
   * These functions display relevant per-iteration information, in a similar fashion to Matlab solvers
   * Iter: iteration count
   * Fval: current energy value
   * Step size: step size used to decrease along the descent direction
   * Optimality: norm of the gradient
   */
  void display_header() const;
  void display_line(scalar step_size);
  void display_status() const;

  scalar energy(Eigen::Ref<const Vec<scalar>> x);
  Vec<scalar> gradient(Eigen::Ref<const Vec<scalar>> x);
  Eigen::SparseMatrix<scalar> hessian(Eigen::Ref<const Vec<scalar>> x);

private:
  Vec<scalar> _var;
  int _iter;
  scalar _energy_val;
  Vec<scalar> _force_val;
  Eigen::SparseMatrix<scalar> _hessian_val;
  StatusType _status;
  scalar _regularization_coeff;

  LinearSolver<scalar> _solver;
  std::function<scalar(Eigen::Ref<const Vec<scalar>>)> _energy_fct;
  std::function<Vec<scalar>(Eigen::Ref<const Vec<scalar>>)> _gradient_fct;
  std::function<Eigen::SparseMatrix<scalar>(Eigen::Ref<const Vec<scalar>>)> _hessian_fct;
};

} // namespace optim
