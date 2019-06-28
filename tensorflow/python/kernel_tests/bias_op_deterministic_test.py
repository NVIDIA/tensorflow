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
"""Functional tests for deterministic BiasAdd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import bias_op_test
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

# BiasAddDeterministicTest = bias_op_test.BiasAddTest

LayerShape = collections.namedtuple('LayerShape',
                                    'batch, height, width, depth, channels')

class BiasAddDeterministicTest(bias_op_test.BiasAddTest):

  # Alternatively, the environment variable can be set before calling
  #   test.main()
  @classmethod
  def setUpClass(cls):
    super(BiasAddDeterministicTest, cls).setUpClass()
    # print("##### RUNNING WITH TF_DETERMINISTIC_OPS = 1 ####")
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"

  def _random_data_op(self, shape):
    # np.random.random_sample can properly interpret either tf.TensorShape or
    # namedtuple as a list.
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)

  def _assert_reproducible(self, operation):
    with self.cached_session(force_gpu=True):
      result_1 = self.evaluate(operation)
      result_2 = self.evaluate(operation)
    self.assertAllEqual(result_1, result_2)

  @test_util.run_cuda_only
  def testGradients(self):
    np.random.seed(1)
    channels = 8
    in_shape = LayerShape(batch=8, height=32, width=32, depth=32, channels=channels)
    bias_shape = (channels)
    in_op = self._random_data_op(in_shape)
    bias_op = self._random_data_op(bias_shape)
    out_op = nn_ops.bias_add(in_op, bias_op, data_format="NHWC")
    bias_gradients_op = gradients_impl.gradients(nn_ops.l2_loss(out_op),
    	                   bias_op, colocate_gradients_with_ops=True)
    self._assert_reproducible(bias_gradients_op)

  def testInputDims(self):
    pass

  def testBiasVec(self):
  	pass

  def testBiasInputsMatch(self):
  	pass


if __name__ == "__main__":
  test.main()
