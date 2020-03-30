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

#include "test_utils.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {


bool ExpectClose(double x, double y, double max_abs_relative_difference) {
    if (std::isinf(x) && std::isinf(y)) {
        EXPECT_EQ(std::signbit(x), std::signbit(y));
        return true;
    }

    if (std::isnan(x) && std::isnan(y)) {
        return true;
    }

    double absolute_difference = fabs(x - y);
    double relative_difference = absolute_difference / std::max(fabs(x), fabs(y));
    if (x == 0 || y == 0) {
        // If x or y is exactly zero, then relative difference doesn't have any
        // meaning. Take the absolute difference instead.
        relative_difference = absolute_difference;
    }


    EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
    return relative_difference <= max_abs_relative_difference;
}

void ExpectArraysCloseUptoScale(int n,
                                const double* p,
                                const double* q,
                                double tol) {
    CHECK_GT(n, 0);
    CHECK(p);
    CHECK(q);

    double p_max = 0;
    double q_max = 0;
    int p_i = 0;
    int q_i = 0;

    for (int i = 0; i < n; ++i) {
        if (std::abs(p[i]) > p_max) {
            p_max = std::abs(p[i]);
            p_i = i;
        }
        if (std::abs(q[i]) > q_max) {
            q_max = std::abs(q[i]);
            q_i = i;
        }
    }

    // If both arrays are all zeros, they are equal up to scale, but
    // for testing purposes, that's more likely to be an error than
    // a desired result.
    CHECK_NE(p_max, 0.0);
    CHECK_NE(q_max, 0.0);

    for (int i = 0; i < n; ++i) {
        double p_norm = p[i] / p[p_i];
        double q_norm = q[i] / q[q_i];

        EXPECT_NEAR(p_norm, q_norm, tol) << "i=" << i;
    }
}

void ExpectArraysClose(int n, const double* p, const double* q, double tol) {
    CHECK_GT(n, 0);
    CHECK(p);
    CHECK(q);

    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(ExpectClose(p[i], q[i], tol)) << "p[" << i << "]" << p[i] << " "
                                                  << "q[" << i << "]" << q[i] << " "
                                                  << "tol: " << tol;
    }
}

std::pair<std::vector<double>, std::vector<double> > EvaluateCostFunction(
    CostFunction* cost_function, double value) {
  auto num_residuals = cost_function->num_residuals();
  auto parameter_block_sizes = cost_function->parameter_block_sizes();
  auto num_parameter_blocks = parameter_block_sizes.size();

  int total_num_parameters = 0;
  for (auto block_size : parameter_block_sizes) {
    total_num_parameters += block_size;
  }

  std::vector<double> params_array(total_num_parameters, value);
  std::vector<double*> params(num_parameter_blocks);
  std::vector<double> residuals(num_residuals, 0);
  std::vector<double> jacobians_array(num_residuals * total_num_parameters, 0);
  std::vector<double*> jacobians(num_parameter_blocks);

  for (int i = 0, k = 0; i < num_parameter_blocks;
       k += parameter_block_sizes[i], ++i) {
    params[i] = &params_array[k];
  }

  for (int i = 0, k = 0; i < num_parameter_blocks;
       k += parameter_block_sizes[i], ++i) {
    jacobians[i] = &jacobians_array[k * num_residuals];
  }

  cost_function->Evaluate(params.data(), residuals.data(), jacobians.data());

  return std::make_pair(residuals, jacobians_array);
}

void CompareCostFunctions(CostFunction* cost_function1,
                          CostFunction* cost_function2,

                          double value,
                          double tol) {
  auto residuals_and_jacobians_1 = EvaluateCostFunction(cost_function1, value);
  auto residuals_and_jacobians_2 = EvaluateCostFunction(cost_function2, value);

  ExpectArraysClose(residuals_and_jacobians_1.first.size(),
                    residuals_and_jacobians_1.first.data(),
                    residuals_and_jacobians_2.first.data(),
                    tol);
  ExpectArraysClose(residuals_and_jacobians_1.second.size(),
                    residuals_and_jacobians_1.second.data(),
                    residuals_and_jacobians_2.second.data(),
                    tol);
}

}  // namespace internal
}  // namespace ceres
