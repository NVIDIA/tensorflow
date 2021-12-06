/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_NVTX_UTILS_H_
#define TENSORFLOW_CORE_FRAMEWORK_NVTX_UTILS_H_

#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/profiler/nvtx_utils.h"

namespace tensorflow {

// Forward declarations.
class OpKernel;
class Tensor;

namespace nvtx {

namespace detail {

string GetNodeExecutionRangeMessageImpl(
    const OpKernel* kernel,
    const gtl::InlinedVector<const Tensor*, 4>& input_tensors);

}  // namespace detail

// Returns a message describing the given OpKernel and its inputs, for use in
// annotating an NVTX range.
// If RangesDetailedEnabled() then the message is formatted as JSON and includes
// detailed information about the inputs and attributes, otherwise the message
// includes only the name and op type. Note that construction of detailed
// messages is significantly more expensive.
template <typename InputContainer, class InputToTensorFunc>
inline string GetNodeExecutionRangeMessage(
    const OpKernel* kernel, size_t num_inputs, const InputContainer& inputs,
    InputToTensorFunc input_to_tensor_fn) {
  gtl::InlinedVector<const Tensor*, 4> input_tensors;
  input_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors.push_back(input_to_tensor_fn(inputs[i]));
  }
  return detail::GetNodeExecutionRangeMessageImpl(kernel, input_tensors);
}

// Convenience subclass for creating a range for an op kernel (does not
// include inputs to the node). This is useful for adding a range around the
// end of an async op (e.g., in the lambda it passes to ThenExecute).
template <class Domain = NullDomain>
class ScopedKernelRangeIfEnabled : public ScopedRangeIfEnabled<Domain> {
 public:
  ScopedKernelRangeIfEnabled(const OpKernel* kernel)
      : ScopedRangeIfEnabled<Domain>(kernel->def().op(), [&]() {
          return nvtx::GetNodeExecutionRangeMessage(
              kernel, /*num_inputs=*/0, /*inputs=*/(int*)nullptr,
              [](int _) -> const Tensor* { return nullptr; });
        }) {}
};

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NVTX_UTILS_H_
