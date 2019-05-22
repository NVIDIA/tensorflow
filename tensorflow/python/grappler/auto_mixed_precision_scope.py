# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Exposes the scope for NVIDIA optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

# follows nvidia_scope as example

_AUTO_MIXED_PRECISION_SCOPE_KEY = ("__auto_mixed_precision_scope",)


class AutoMixedPrecisionScope(object):
  """Keeps track of scopes"""

  def __init__(self, depth):
    self.depth = depth


def ConditionalWrapper(selector, value):
  """ Simple wrapper that returns a callable to
      disable attributes for the excluded nodes."""

  def ConditionalSetter(node_def):
    if callable(selector):
      accept = selector(node_def).b
    else:
      accept = selector.b
    if accept:
      return value
    return None

  return ConditionalSetter


@contextlib.contextmanager
def auto_mixed_precision_scope(
    select=True,
    scope_name=None,
):
  """Include or exclude nodes in optimized segment with given parameters.

  NOTE: this is a hint and will be supported on a best-effort basis.
  Args:
    select       : Whether to include the node or not.
      Can be a callable. (Default= True)
    scope_name   : Name of the scope. Same named scopes will attempted to be
      merged, if they have identical parameters. If None, will be generated
      during conversion (Default=None)

  Yields:
    The current scope, enabling or disabling inclusion.

  """
  # enum would have been better here but we need attrvalue to have these enums
  # which is not possible. So we stick with strings.

  if callable(select):

    def segment_include(node_def):
      return attr_value_pb2.AttrValue(b=select(node_def))
  else:
    segment_include = attr_value_pb2.AttrValue(b=select)

  attrs = {"_AutoMixedPrecisionSegmentInclude": segment_include}

  if scope_name:
    attrs["_AutoMixedPrecisionSegmentName"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(s=scope_name.encode()))


  # Find the singleton counter for the current scoped graph.  If it
  # doesn't exist, create one.
  auto_mixed_precision_scope_counter = ops.get_collection(
                           _AUTO_MIXED_PRECISION_SCOPE_KEY)
  if not auto_mixed_precision_scope_counter:
    auto_mixed_precision_scope_counter = AutoMixedPrecisionScope(0)
    ops.add_to_collection(_AUTO_MIXED_PRECISION_SCOPE_KEY,
                          auto_mixed_precision_scope_counter)
  else:
    auto_mixed_precision_scope_counter = auto_mixed_precision_scope_counter[0]

  auto_mixed_precision_scope_counter.depth += 1

  # Add depth of scope as an attribute accessible by optimizer
  attrs["_AutoMixedPrecisionSegmentDepth"] = attr_value_pb2.AttrValue(
                            i=auto_mixed_precision_scope_counter.depth)

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

  auto_mixed_precision_scope_counter.depth -= 1

