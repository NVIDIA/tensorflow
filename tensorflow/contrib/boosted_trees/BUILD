# TensorFlow code for training gradient boosted trees.

load("//tensorflow:tensorflow.bzl", "py_test", "tf_custom_op_library", "tf_gen_op_libs", "tf_gen_op_wrapper_py", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core/platform:default/build_config_root.bzl", "if_dynamic_pywrap")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

package_group(name = "friends")

cc_library(
    name = "boosted_trees_kernels",
    deps = [
        ":model_ops_kernels",
        ":prediction_ops_kernels",
        ":quantile_ops_kernels",
        ":split_handler_ops_kernels",
        ":stats_accumulator_ops_kernels",
        ":training_ops_kernels",
    ],
    alwayslink = 1,
)

cc_library(
    name = "boosted_trees_ops_op_lib",
    deps = [
        ":model_ops_op_lib",
        ":prediction_ops_op_lib",
        ":quantile_ops_op_lib",
        ":split_handler_ops_op_lib",
        ":stats_accumulator_ops_op_lib",
        ":training_ops_op_lib",
    ],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_py",
        ":losses",
    ],
)

py_library(
    name = "losses",
    srcs = ["python/utils/losses.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
    ],
)

py_test(
    name = "losses_test",
    size = "small",
    srcs = ["python/utils/losses_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "gbdt_batch",
    srcs = [
        "python/training/functions/gbdt_batch.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_model_ops_py",
        "//tensorflow/contrib/boosted_trees:batch_ops_utils_py",
        "//tensorflow/contrib/boosted_trees:boosted_trees_ops_py",
        "//tensorflow/contrib/boosted_trees/lib:categorical_split_handler",
        "//tensorflow/contrib/boosted_trees/lib:ordinal_split_handler",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
    ],
)

py_test(
    name = "gbdt_batch_test",
    size = "medium",
    srcs = ["python/training/functions/gbdt_batch_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "nofwdcompat",  # b/137641346
        "notsan",  # b/62863147
    ],
    deps = [
        ":gbdt_batch",
        ":losses",
        ":model_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
    ],
)

# Kernel tests

py_test(
    name = "model_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/model_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":model_ops_py",
        ":prediction_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "prediction_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/prediction_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":model_ops_py",
        ":prediction_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "quantile_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/quantile_ops_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":quantile_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "split_handler_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/split_handler_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":split_handler_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "stats_accumulator_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/stats_accumulator_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":stats_accumulator_ops_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
    ],
)

py_test(
    name = "training_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/training_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":model_ops_py",
        ":training_ops_py",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//third_party/py/numpy",
    ],
)

# Ops

py_library(
    name = "batch_ops_utils_py",
    srcs = ["python/ops/batch_ops_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

tf_custom_op_py_library(
    name = "boosted_trees_ops_loader",
    srcs = ["python/ops/boosted_trees_ops_loader.py"],
    dso = [":python/ops/_boosted_trees_ops.so"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "boosted_trees_ops_py",
    srcs_version = "PY2AND3",
    deps = [
        ":model_ops_py",
        ":prediction_ops_py",
        ":quantile_ops_py",
        ":split_handler_ops_py",
        ":stats_accumulator_ops_py",
        ":training_ops_py",
    ],
)

# Model Ops.
tf_gen_op_libs(
    op_lib_names = ["model_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_model_ops_py",
    out = "python/ops/gen_model_ops.py",
    deps = [":model_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "model_ops_py",
    srcs = ["python/ops/model_ops.py"],
    kernels = [
        ":model_ops_kernels",
        ":model_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_loader",
        ":gen_model_ops_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
    ],
)

tf_kernel_library(
    name = "model_ops_kernels",
    srcs = ["kernels/model_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_boosted_trees_ops.so",
    srcs = [
        "kernels/model_ops.cc",
        "kernels/prediction_ops.cc",
        "kernels/quantile_ops.cc",
        "kernels/split_handler_ops.cc",
        "kernels/stats_accumulator_ops.cc",
        "kernels/training_ops.cc",
        "ops/model_ops.cc",
        "ops/prediction_ops.cc",
        "ops/quantile_ops.cc",
        "ops/split_handler_ops.cc",
        "ops/stats_accumulator_ops.cc",
        "ops/training_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:example_partitioner",
        "//tensorflow/contrib/boosted_trees/lib:models",
        "//tensorflow/contrib/boosted_trees/lib:node-stats",
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/lib:weighted_quantiles",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/contrib/boosted_trees/resources:quantile_stream_resource",
        "//tensorflow/contrib/boosted_trees/resources:stamped_resource",
    ] + if_dynamic_pywrap([
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc_headers_only",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc_headers_only",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_cc_headers_only",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc_headers_only",
    ], otherwise = [
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
    ]),
)

# Split handler Ops.
tf_gen_op_libs(
    op_lib_names = ["split_handler_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_split_handler_ops_py",
    out = "python/ops/gen_split_handler_ops.py",
    deps = [
        ":split_handler_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "split_handler_ops_py",
    srcs = ["python/ops/split_handler_ops.py"],
    kernels = [
        ":split_handler_ops_kernels",
        ":split_handler_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_loader",
        ":gen_split_handler_ops_py",
    ],
)

tf_kernel_library(
    name = "split_handler_ops_kernels",
    srcs = ["kernels/split_handler_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:node-stats",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Training Ops.
tf_gen_op_libs(
    op_lib_names = [
        "training_ops",
    ],
    deps = ["//tensorflow/contrib/boosted_trees/proto:learner_proto_cc"],
)

tf_gen_op_wrapper_py(
    name = "gen_training_ops_py",
    out = "python/ops/gen_training_ops.py",
    deps = [
        ":training_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "training_ops_py",
    srcs = ["python/ops/training_ops.py"],
    kernels = [
        ":training_ops_kernels",
        ":training_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_loader",
        ":gen_training_ops_py",
    ],
)

tf_kernel_library(
    name = "training_ops_kernels",
    srcs = ["kernels/training_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/lib:weighted_quantiles",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/contrib/boosted_trees/resources:quantile_stream_resource",
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

# Prediction Ops.
tf_gen_op_libs(
    op_lib_names = ["prediction_ops"],
    deps = ["//tensorflow/contrib/boosted_trees/proto:learner_proto_cc"],
)

tf_gen_op_wrapper_py(
    name = "gen_prediction_ops_py",
    out = "python/ops/gen_prediction_ops.py",
    deps = [
        ":prediction_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "prediction_ops_py",
    srcs = ["python/ops/prediction_ops.py"],
    kernels = [
        ":prediction_ops_kernels",
        ":prediction_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_loader",
        ":gen_prediction_ops_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_kernel_library(
    name = "prediction_ops_kernels",
    srcs = ["kernels/prediction_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:example_partitioner",
        "//tensorflow/contrib/boosted_trees/lib:models",
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Quantile ops
tf_gen_op_libs(
    op_lib_names = ["quantile_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_quantile_ops_py_wrap",
    out = "python/ops/gen_quantile_ops.py",
    deps = [
        ":quantile_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "quantile_ops_py",
    srcs = ["python/ops/quantile_ops.py"],
    kernels = [
        ":quantile_ops_kernels",
        ":quantile_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batch_ops_utils_py",
        ":boosted_trees_ops_loader",
        ":gen_quantile_ops_py_wrap",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
    ],
)

tf_kernel_library(
    name = "quantile_ops_kernels",
    srcs = ["kernels/quantile_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/lib:weighted_quantiles",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/contrib/boosted_trees/resources:quantile_stream_resource",
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

# Stats Accumulator ops
tf_gen_op_libs(
    op_lib_names = ["stats_accumulator_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_stats_accumulator_ops_py_wrap",
    out = "python/ops/gen_stats_accumulator_ops.py",
    deps = [
        ":stats_accumulator_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "stats_accumulator_ops_py",
    srcs = ["python/ops/stats_accumulator_ops.py"],
    kernels = [
        ":stats_accumulator_ops_kernels",
        ":stats_accumulator_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batch_ops_utils_py",
        ":boosted_trees_ops_loader",
        ":gen_stats_accumulator_ops_py_wrap",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
    ],
)

tf_kernel_library(
    name = "stats_accumulator_ops_kernels",
    srcs = ["kernels/stats_accumulator_ops.cc"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/resources:stamped_resource",
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

# Pip

py_library(
    name = "boosted_trees_pip",
    deps = [
        ":init_py",
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/boosted_trees/estimator_batch:custom_export_strategy",
        "//tensorflow/contrib/boosted_trees/estimator_batch:dnn_tree_combined_estimator",
        "//tensorflow/contrib/boosted_trees/estimator_batch:init_py",
        "//tensorflow/contrib/boosted_trees/estimator_batch:trainer_hooks",
        "//tensorflow/contrib/boosted_trees/lib:categorical_split_handler",
        "//tensorflow/contrib/boosted_trees/lib:ordinal_split_handler",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
    ],
)
