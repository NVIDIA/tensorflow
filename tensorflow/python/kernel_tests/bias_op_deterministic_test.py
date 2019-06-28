# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for BiasAdd when operating deterministically."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import bias_op_test
from tensorflow.python.platform import test

class BiasAddDeterministicTest(bias_op_test.BiasAddTest):

  # TODO(duncanriach): test that the bias gradients are deterministic
  @test_util.run_cuda_only
  def testBiasGradients(self):
    pass

  # TODO(duncanriach): implement the error checks and enable the following
  #                    three tests by not overriding them
  def testInputDims(self):
    pass

  def testBiasVec(self):
    pass

  def testBiasInputsMatch(self):
    pass


if __name__ == "__main__":
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  test.main()
