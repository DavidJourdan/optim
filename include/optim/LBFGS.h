// LBFGS.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 06/15/20

#ifndef LBFGS_H
#define LBFGS_H

#include "SolverBase.h"

template <typename scalar>
using Mat = Eigen::Matrix<scalar, -1, -1, Eigen::ColMajor>;

namespace optim
{

template <typename scalar = double>
class LBFGSSolver : public SolverBase<scalar>
{
public:
  /**
   * @brief Construct a new LBFGS Solver object, you can either initialize it by calling init
   * and then call solve_one_step() as many time as you want, or call directly solve
   */
  LBFGSSolver() : SolverBase<scalar>::SolverBase{} {};

  /**
   * @brief Initialize the solver. Set the energy and gradient functions
   *
   * @param objective_func  function to minimize (takes as input an Eigen::VectorXd of size n)
   * @param gradient_func  gradient of the objective function (returns an Eigen::VectorXd of size n)
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   */
  template <class ScalarFunc, class VectorFunc>
  void init(const ScalarFunc &objective_func, const VectorFunc &gradient_func, Eigen::Ref<const Vec<scalar>> var)
  {
    _iter = 1;

    _energy_fct = objective_func;
    _gradient_fct = gradient_func;
    _var = var;
    _grad = gradient(var);

    this->init_base(energy(var));

    this->options.update_fct(var);

    int n = var.size();
    _s.resize(n, this->options.bfgs.m);
    _y.resize(n, this->options.bfgs.m);
    _ys.resize(this->options.bfgs.m);
    _oldVar = _var;
    _oldGrad = _grad;
  }

  /**
   * @brief Set the energy and gradient using a class implementing these functions
   *
   * @tparam Obj any type with the methods energy, gradient and hessian
   * @param obj  an object implementing the objective function and its first derivative
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   */
  template <class Obj>
  void init(Obj &obj, Eigen::Ref<const Vec<scalar>> var)
  {
    init([&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.energy(x); },
         [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.gradient(x); }, var);
  }

  /**
   * @brief  solve the full nonlinear optimization problem in an automated manner
   * (look at BFGSOptions to see how to fine tune the behavior of this method)
   * The algorithm will stop when the gradient is close enough to 0 ("close enough" is defined by options.threshold)
   * @param objective_func  function to minimize (takes as input an Eigen::VectorXd of size n)
   * @param gradient_func  gradient of the objective function (returns an Eigen::VectorXd of size n)
   * @param _var  initial guess for the algorithm (Eigen::VectorXd of size n)
   * @return the result of the algorithm (ie x = \argmin f such that \nabla f(x) = 0 and \nabla^2 f(x) is SPD)
   */
  template <class ScalarFunc, class VectorFunc>
  Vec<scalar>
  solve(const ScalarFunc &objective_func, const VectorFunc &gradient_func, Eigen::Ref<const Vec<scalar>> var)
  {
    init(objective_func, gradient_func, var);

    this->main_loop();

    return _var;
  }

  /**
   * @brief  solve the full nonlinear optimization problem in an automated manner, overload for objects implementing the
   * needed functions
   *
   * @tparam Obj any type with the methods energy, gradient and hessian
   * @param obj  an object implementing the objective function and its first derivative
   * @param var  initial guess for the algorithm (Eigen::VectorXd of size n)
   * @return the result of the algorithm (ie x = \argmin f such that \nabla f(x) = 0 and \nabla^2 f(x) is SPD)
   */
  template <class Obj>
  Vec<scalar> solve(Obj &obj, Eigen::Ref<const Vec<scalar>> var)
  {
    return solve([&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.energy(x); },
                 [&obj](Eigen::Ref<const Vec<scalar>> x) { return obj.gradient(x); }, var);
  }

  /**
   * @brief  do one step of the algorithm
   *
   * @return the result of doing one step of the algorithm (ie x_{k+1} = x_k - \alpha * dx where dx is the computed
   * descent direction)
   */
  Vec<scalar> solve_one_step() override;

  const char *method_name() const override { return "LIMITED-MEMORY BFGS (LBFGS)"; }

  // getters
  int iteration_count() const override { return _iter; }
  scalar gradient_norm() const override { return _grad.norm(); }
  Vec<scalar> var() const override { return _var; }
  Vec<scalar> gradient_value() const override { return _grad; }

protected:
  scalar energy(Eigen::Ref<const Vec<scalar>> x) override;
  Vec<scalar> gradient(Eigen::Ref<const Vec<scalar>> x);

  const auto y_col(int k) const { return _y.col(k % this->options.bfgs.m); }
  const auto s_col(int k) const { return _s.col(k % this->options.bfgs.m); }
  const scalar y_dot_s(int k) const { return _ys(k % this->options.bfgs.m); }

  void update_s(int k) { _s.col(k % this->options.bfgs.m) = _var - _oldVar; }
  void update_y(int k) { _y.col(k % this->options.bfgs.m) = _grad - _oldGrad; }
  void compute_y_dot_s(int k) { _ys(k % this->options.bfgs.m) = y_col(k).dot(s_col(k)); }

private:
  Mat<scalar> _s;  // n by m matrix of secants: s_{k+1} = x_{k+1} - x_k
  Mat<scalar> _y;  // n by m matrix of gradient differences: y_{k+1} = g_{k+1} - g_k
  Vec<scalar> _ys; // Vector of size m, coloum-wise dot products between _s and _y
  int _iter;       // number of iterations

  Vec<scalar> _oldVar;
  Vec<scalar> _var;
  Vec<scalar> _oldGrad;
  Vec<scalar> _grad;

  std::function<scalar(Eigen::Ref<const Vec<scalar>>)> _energy_fct;
  std::function<Vec<scalar>(Eigen::Ref<const Vec<scalar>>)> _gradient_fct;
};

} // namespace optim

#endif // LBFGS_H
