// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
//
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_COST_FUNCTION_TO_FUNCTOR_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_COST_FUNCTION_TO_FUNCTOR_H_

#include "AutodiffCodegen/macros.h"

namespace ceres {
namespace internal {


// CodegenCostFunctions have both, an templated operator() and the Evaluate()
// function. The operator() is used during code generation and Evaluate() is
// used during execution.
//
// If we want to use the operator() during execution (with autodiff) this
// wrapper class here has to be used. Autodiff doesn't support functors that
// have an Evaluate() function.
//
// CostFunctionToFunctor hides the Evaluate() function, because it doesn't
// derive from CostFunction. Autodiff sees it as a simple functor and will use
// the operator() as expected.
template <typename CostFunction>
struct CostFunctionToFunctor {
    template <typename... _Args>
    CostFunctionToFunctor(_Args&&... __args)
        : cost_function(std::forward<_Args>(__args)...) {}

    template <typename... _Args>
    bool operator()(_Args&&... __args) const {
        return cost_function(std::forward<_Args>(__args)...);
    }

    CostFunction cost_function;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_COST_FUNCTION_TO_FUNCTOR_H_
