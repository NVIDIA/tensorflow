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
// #include <deque>
// #include <memory>
// #include <unordered_set>
// #include <vector>
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NODE_SHAPE_INFO_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NODE_SHAPE_INFO_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

static bool ExcludePossibleDynamicOps() {
  static bool exclude = [] {
    bool to_be_excluded = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_XLA_DO_NOT_COMPILE_POSSIBLE_DYNAMIC_OPS",
        /*default_val=*/false, &to_be_excluded));
    return to_be_excluded;
  }();
  return exclude;
}

class NodeShapesInfo {
 public:
  static NodeShapesInfo* GetNodeShapesInfo() {
    static NodeShapesInfo* self = new NodeShapesInfo();
    return self;
  }

  std::pair<bool, string> GetShapeDefinition(string node_name, int input_id) {
    auto null_pair = std::make_pair(false, "");
    if (node_name_shape_def_.find(node_name) == node_name_shape_def_.end()) {
      LOG(WARNING) << "Relevant shape information for " << node_name
                   << " has not been stored yet.";
      return null_pair;
    }
    auto in_id_vec = node_name_shape_def_[node_name];
    if (in_id_vec.empty()) {
      LOG(WARNING)
          << "Entry found for " << node_name
          << " but shape information is not populated in the singleton.";
      return null_pair;
    }
    if (input_id > in_id_vec.size() - 1) {
      LOG(WARNING)
          << "The input id requested i.e, " << input_id
          << " is greater that the size of stored shape inference data. The "
             "number inputs stored with shape inference data is "
          << in_id_vec.size();
      return null_pair;
    }

    VLOG(1) << "node name: " << node_name << " num_inputs: " << in_id_vec.size()
            << " in_id requested = " << input_id;
    return in_id_vec[input_id];
  }

  void SetShapeDefinition(string node_name, bool is_shape_fully_defined,
                          string shape_def, int in_id) {
    if (node_name_shape_def_.find(node_name) == node_name_shape_def_.end()) {
      std::vector<std::pair<bool, string>> p;
      p.push_back(std::make_pair(is_shape_fully_defined, shape_def));
      node_name_shape_def_[node_name] = p;
    } else {
      node_name_shape_def_[node_name].push_back(
          std::make_pair(is_shape_fully_defined, shape_def));
    }
  }

  void InitializeFullyDefinedNodeMapVector(string node_name, int n) {
    fully_defined_nodes_[node_name].resize(n);
  }

  void AddFullyDefinedNode(string node_name, string shape_def, int id) {
    CHECK_LT(id, fully_defined_nodes_[node_name].size());
    fully_defined_nodes_[node_name][id] = shape_def;
  }

  bool IsNodeFullyDefined(string node_name) {
    return fully_defined_nodes_.count(node_name);
  }

  absl::optional<std::vector<string>> GetOutEdgesOfFullyDefinedNode(
      string node_name) {
    if (!fully_defined_nodes_.count(node_name)) return absl::nullopt;
    return fully_defined_nodes_[node_name];
  }

  string ToStringFullyDefinedNode(string node_name) {
    CHECK_EQ(IsNodeFullyDefined(node_name), true);
    string result;
    // For each consumer node of 'node_name', string representation of shape
    // for all the outputs.
    absl::StrAppend(&result, " [",
                    absl::StrJoin(fully_defined_nodes_[node_name], ","), "]");
    return result;
  }

  string ToStringFullyDefinedNodes() {
    string result;
    // For each fully defined node, string representation of shape for all the
    // outputs.
    for (const auto& a : fully_defined_nodes_) {
      absl::StrAppend(&result, a.first, " ");
      std::vector<std::pair<bool, string>> inputs =
          node_name_shape_def_[a.first];
      for (const auto& b : inputs) {
        absl::StrAppend(&result, b.second);
      }
      absl::StrAppend(&result, " -> [", absl::StrJoin(a.second, ","), "]");
      absl::StrAppend(&result, "\n");
    }
    return result;
  }

  void SetFullyDefinedNodesOutShapesVec(string node_name,
                                        std::vector<string>&& shape_vec) {
    fully_defined_nodes_[node_name] = shape_vec;
  }

 private:
  explicit NodeShapesInfo() {}

  // Maps unique node name in graph def to vector of pairs indexed by input id.
  // The pair of bool indicating whether shape is fully defined along with a
  // string describing the shape.
  std::unordered_map<string, std::vector<std::pair<bool, string>>>
      node_name_shape_def_;

  // Map of node name with all its shapes fully defined based on shape
  // inference. The node name maps to a vector of shape strings sorted by
  // src_output idx of the node.
  std::unordered_map<string, std::vector<string>> fully_defined_nodes_;

  ~NodeShapesInfo() {}
  TF_DISALLOW_COPY_AND_ASSIGN(NodeShapesInfo);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NODE_SHAPE_INFO_H_