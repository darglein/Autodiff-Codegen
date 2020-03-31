# Autodiff-Codegen

Autodiff-Codegen is a C++ code generator for C++ cost functors. 
The output is a header file with the jacobian implementation of the functor. 
This can be used, for example, for solving non-linear least-squares problems.
The output function matches the cost-function-interface defined by the Ceres-Solver.

Advantages
- Higher performance compared to direct autodiff (GCC, MSVC)
- C++ code of Jacobian code, which can inspected and further optimized

### Ceres Integration

AutodiffCodegen was included in Ceres until around April, 2020. 
It showed an improvement of around 2x-5x compared to the old autodiff.
Using AutodiffCodegen as a baseline, we were then able to improve the autodiff performance to the same level (only for the Clang compiler).
AutodiffCodegen was therefore not needed anymore.
