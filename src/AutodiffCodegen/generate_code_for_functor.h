// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
//
#ifndef CERES_PUBLIC_CODEGEN_AUTODIFF_H_
#define CERES_PUBLIC_CODEGEN_AUTODIFF_H_

#include "AutodiffCodegen/internal/code_generator.h"
#include "AutodiffCodegen/internal/expression_graph.h"
#include "AutodiffCodegen/internal/expression_ref.h"
#include "AutodiffCodegen/ceres/internal/autodiff.h"
#include "AutodiffCodegen/ceres/jet.h"

namespace ceres {

struct AutoDiffCodeGenOptions {};

// TODO(darius): Documentation
template <typename DerivedCostFunctor>
std::vector<std::string> GenerateCodeForFunctor(
    const AutoDiffCodeGenOptions& options) {
  // Define some types and shortcuts to make the code below more readable.
  using ParameterDims = typename DerivedCostFunctor::ParameterDims;
  using Parameters = typename ParameterDims::Parameters;
  // Instead of using scalar Jets, we use Jets of ExpressionRef which record
  // their own operations during evaluation.
  using ExpressionRef = internal::ExpressionRef;
  using ExprJet = Jet<ExpressionRef, ParameterDims::kNumParameters>;
  constexpr int kNumResiduals = DerivedCostFunctor::kNumResiduals;
  constexpr int kNumParameters = ParameterDims::kNumParameters;
  constexpr int kNumParameterBlocks = ParameterDims::kNumParameterBlocks;

  // Create the cost functor using the default constructor.
  // Code is generated for the CostFunctor and not an instantiation of it. This
  // is different to AutoDiffCostFunction, which computes the derivatives for
  // a specific object.
  static_assert(std::is_default_constructible<DerivedCostFunctor>::value,
                "Cost functors used in code generation must have a default "
                "constructor. If you are using local variables, make sure to "
                "wrap them into the CERES_LOCAL_VARIABLE macro.");
  DerivedCostFunctor functor;

  // During recording phase all operations on ExpressionRefs are recorded to an
  // internal data structure, the ExpressionGraph. This ExpressionGraph is then
  // optimized and converted back into C++ code.
  internal::StartRecordingExpressions();

  // The Jet arrays are defined after StartRecordingExpressions, because Jets
  // are zero-initialized in the default constructor. This already creates
  // COMPILE_TIME_CONSTANT expressions.
  std::array<ExprJet, kNumParameters> all_parameters;
  std::array<ExprJet, kNumResiduals> residuals;
  std::array<ExprJet*, kNumParameterBlocks> unpacked_parameters =
      ParameterDims::GetUnpackedParameters(all_parameters.data());

  // Create input expressions that convert from the doubles passed from Ceres
  // into codegen Expressions. These inputs are assigned to the scalar part "a"
  // of the corresponding Jets.
  //
  // Example code generated by these expressions:
  //   v_0 = parameters[0][0];
  //   v_1 = parameters[0][1];
  //   ...
  for (int i = 0; i < kNumParameterBlocks; ++i) {
    for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
      ExprJet& parameter = unpacked_parameters[i][j];
      parameter.a = internal::MakeInputAssignment<ExpressionRef>(
          0.0,
          ("parameters[" + std::to_string(i) + "][" + std::to_string(j) + "]")
              .c_str());
    }
  }

  // During the array initialization above, the derivative part of the Jets is
  // set to zero. Here, we set the correct element to 1.
  for (int i = 0; i < kNumParameters; ++i) {
    all_parameters[i].v(i) = ExpressionRef(1);
  }

  // Run the cost functor with Jets of ExpressionRefs.
  // Since we are still in recording mode, all operations of the cost functor
  // will be added to the graph.
  internal::VariadicEvaluate<ParameterDims>(
      functor, unpacked_parameters.data(), residuals.data());

  // At this point the Jets in 'residuals' contain references to the output
  // expressions. Here we add new expressions that assign the generated
  // temporaries to the actual residual array.
  //
  // Example code generated by these expressions:
  //    residuals[0] = v_200;
  //    residuals[1] = v_201;
  //    ...
  for (int i = 0; i < kNumResiduals; ++i) {
    auto& J = residuals[i];
    // Note: MakeOutput automatically adds the expression to the active graph.
    internal::MakeOutput(J.a, "residuals[" + std::to_string(i) + "]");
  }

  // Example code generated by these expressions:
  //    jacobians[0][0] = v_351;
  //    jacobians[0][1] = v_352;
  //    ...
  for (int i = 0, total_param_id = 0; i < kNumParameterBlocks;
       total_param_id += ParameterDims::GetDim(i), ++i) {
    for (int r = 0; r < kNumResiduals; ++r) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        internal::MakeOutput(
            (residuals[r].v[total_param_id + j]),
            "jacobians[" + std::to_string(i) + "][" +
                std::to_string(r * ParameterDims::GetDim(i) + j) + "]");
      }
    }
  }

  // Stop recording and return the current active graph. Performing operations
  // of ExpressionRef after this line will result in an error.
  auto residual_and_jacobian_graph = internal::StopRecordingExpressions();

  // TODO(darius): Once the optimizer is in place, call it from
  // here to optimize the code before generating.

  // We have the optimized code of the cost functor stored in the
  // ExpressionGraphs. Now we generate C++ code for it and place it line-by-line
  // in this vector of strings.
  std::vector<std::string> output;

  output.emplace_back("// This file is generated with ceres::AutoDiffCodeGen.");
  output.emplace_back("// http://ceres-solver.org/");
  output.emplace_back("");

  {
    // Generate C++ code for the EvaluateResidualAndJacobian function and append
    // it to the output.
    internal::CodeGenerator::Options generator_options;
    generator_options.function_name =
        "void EvaluateResidualAndJacobian(double const* const* parameters, "
        "double* "
        "residuals, double** jacobians) const";
    internal::CodeGenerator gen(residual_and_jacobian_graph, generator_options);
    std::vector<std::string> code = gen.Generate();
    output.insert(output.end(), code.begin(), code.end());
  }

  output.emplace_back("");

  // Generate a generic combined function, which calls EvaluateResidual and
  // EvaluateResidualAndJacobian. This combined function is compatible to
  // CostFunction::Evaluate. Therefore the generated code can be directly used
  // in SizedCostFunctions.
  output.emplace_back("bool Evaluate(double const* const* parameters,");
  output.emplace_back("              double* residuals,");
  output.emplace_back("              double** jacobians) const {");

  output.emplace_back("  if (!jacobians) {");
  output.emplace_back("    // Use the input cost functor");
  output.emplace_back("    return (*this)(");
  for (int i = 0; i < kNumParameterBlocks; ++i) {
    output.emplace_back("      parameters[" + std::to_string(i) + "],");
  }
  output.emplace_back("      residuals");
  output.emplace_back("    );");
  output.emplace_back("  }");

  // Create a tmp array of all jacobians and use it for evaluation if the input
  // jacobian is null. The generated code for a <2,3,1,2> cost functor is:
  //   double jacobians_data[6];
  //   double* jacobians_ptrs[] = {
  //       jacobians[0] ? jacobians[0] : jacobians_data + 0,
  //       jacobians[1] ? jacobians[1] : jacobians_data + 6,
  //       jacobians[2] ? jacobians[2] : jacobians_data + 8,
  //   };
  output.emplace_back("  double jacobians_data[" +
                      std::to_string(kNumParameters * kNumResiduals) + "];");
  output.emplace_back("  double* jacobians_ptrs[] = {");
  for (int i = 0, total_param_id = 0; i < kNumParameterBlocks;
       total_param_id += ParameterDims::GetDim(i), ++i) {
    output.emplace_back("    jacobians[" + std::to_string(i) +
                        "] ? jacobians[" + std::to_string(i) +
                        "] : jacobians_data + " +
                        std::to_string(kNumResiduals * total_param_id) + ",");
  }
  output.emplace_back("  };");
  output.emplace_back(
      "  EvaluateResidualAndJacobian(parameters, residuals, "
      "jacobians_ptrs);");

  output.emplace_back("  return true;");
  output.emplace_back("}");

  return output;
}

}  // namespace ceres
#endif  // CERES_PUBLIC_CODEGEN_AUTODIFF_H_
