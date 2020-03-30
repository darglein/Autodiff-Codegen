// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: darius.rueckert@fau.de (Darius Rueckert)

#ifndef CERES_INTERNAL_CODEGEN_TEST_UTILS_H_
#define CERES_INTERNAL_CODEGEN_TEST_UTILS_H_

#include "AutodiffCodegen/ceres/internal/autodiff.h"
#include "AutodiffCodegen/ceres/sized_cost_function.h"

namespace ceres {
namespace internal {


// Expects that x and y have a relative difference of no more than
// max_abs_relative_difference. If either x or y is zero, then the relative
// difference is interpreted as an absolute difference.
//
// If x and y have the same non-finite value (inf or nan) we treat them as being
// close. In such a case no error is thrown and true is returned.
bool ExpectClose(double x, double y, double max_abs_relative_difference);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] - q[i]| / max(|p[i]|, |q[i]|) < tolerance
void ExpectArraysClose(int n,
                       const double* p,
                       const double* q,
                       double tolerance);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] / max_norm_p - q[i] / max_norm_q| < tolerance
//
// where max_norm_p and max_norm_q are the max norms of the arrays p
// and q respectively.
void ExpectArraysCloseUptoScale(int n,
                                const double* p,
                                const double* q,
                                double tolerance);


// Evaluate a cost function and return the residuals and jacobians.
// All parameters are set to 'value'.
std::pair<std::vector<double>, std::vector<double>> EvaluateCostFunction(
    CostFunction* cost_function, double value);

// Evaluates the two cost functions using the method above and then compares the
// result. The comparison uses GTEST expect macros so this function should be
// called from a test enviroment.
void CompareCostFunctions(CostFunction* cost_function1,
                          CostFunction* cost_function2,
                          double value,
                          double tol);

}  // namespace internal
}  // namespace ceres

#endif
