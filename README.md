# Optim
[![](https://github.com/DavidJourdan/optim/workflows/Build/badge.svg)](https://github.com/DavidJourdan/optim/actions)

Nonlinear optimization library built using Eigen

## Newton solver

The Newton Solver is meant to work with medium-sized problems and sparse hessian matrices. To use this algorithm, you need to give both first-order (gradient) and second-order information (hessian). Here is an example with the Rosenbrock function

```cpp
using namespace Eigen;
struct Rosenbrock {
    int n;
    Rosenbrock(int n_) : n(n_) {}
    double energy(const Ref<const VectorXd> x)
    {
        double res = 0;
        for(int i = 0; i < n; i += 2)
        {
          res += pow(1.0 - x(i), 2) + 100 * pow(x(i + 1) - x(i) * x(i), 2);
        }
        return res;
    }
    VectorXd gradient(const Ref<const VectorXd> x)
    {
        VectorXd grad = VectorXd::Zero(n);
        for(int i = 0; i < n; i += 2)
        {
          grad(i + 1) = 200 * (x(i + 1) - x(i) * x(i));
          grad(i) = -2 * (x(i) * grad(i + 1) + 1.0 - x(i));
        }
        return grad;
    }
    SparseMatrix<double> hessian(const Ref<const VectorXd> x)
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
```

Then you can either solve the problem automatically:
```cpp
using namespace optim;
NewtonSolver<double> solver;
solver.options.display = NewtonSolver<double>::quiet;
solver.options.threshold = 1e-4; // if not set, the threshold will be given a default value

Rosenbrock r(10);
VectorXd x = VectorXd::Constant(10, 3);
solver.solve(r, x);
```

Or, if you want more control at each iteration, you can solve one step at a time:
```cpp
using namespace optim;
NewtonSolver<double> solver;
solver.options.display = NewtonSolver<double>::verbose;

Rosenbrock r(10);
VectorXd x = VectorXd::Constant(10, 3);
solver.init(r, x);
while(solver.gradient_norm() > 1e-4 && solver.info() == NewtonSolver<double>::success)
{
    solver.solve_one_step();
}
```
Note that you can also initialize the solver with the energy, gradient and hessian function themselves instead of having a specific class:
```cpp
VectorXd var = VectorXd::Constant(10, 3);
solver.solve([&r](Eigen::Ref<const Vec<scalar>> x) { return r.energy(x); },
      [&r](Eigen::Ref<const Vec<scalar>> x) { return r.gradient(x); },
      [&r](Eigen::Ref<const Vec<scalar>> x) { return r.hessian(x); }, var);
```

The implementation of this algorithm follow its description given in "Nonlinear Optimization" by Nocedal & Wright

## How to build

If your project uses CMake, simply add 
```
add_subdirectory("path/to/optim")
// ...
target_link_libraries(YOUR_TARGET optim)
```
to your ```CMakeLists.txt```. You can optionally add CHOLMOD as a dependency for faster matrix solves, to do so set the ```OPTIM_USE_CHOLMOD``` to ```ON``` in your CMakeLists.txt
