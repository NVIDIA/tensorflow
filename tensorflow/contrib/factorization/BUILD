# Description:
# Contains ops for factorization of data, including matrix factorization and
# clustering.

load("//tensorflow:tensorflow.bzl", "py_test", "tf_custom_op_library", "tf_gen_op_libs", "tf_gen_op_wrapper_py", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core/platform:default/build_config_root.bzl", "if_dynamic_pywrap")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_custom_op_py_library(
    name = "factorization_py",
    srcs = [
        "__init__.py",
        "python/ops/clustering_ops.py",
        "python/ops/factorization_ops.py",
        "python/ops/gmm.py",
        "python/ops/gmm_ops.py",
        "python/ops/kmeans.py",
        "python/ops/wals.py",
    ],
    dso = [
        ":python/ops/_factorization_ops.so",
    ],
    kernels = [
        ":all_ops",
        "//tensorflow/contrib/factorization/kernels:all_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":factorization_ops_test_utils_py",
        ":gen_factorization_ops",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clustering_ops_gen",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column:feature_column_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
    deps = [
        "//tensorflow/contrib/learn",
    ],
)

tf_custom_op_library(
    name = "python/ops/_factorization_ops.so",
    srcs = [
        "ops/factorization_ops.cc",
    ],
    deps = if_dynamic_pywrap([], otherwise = [
        "//tensorflow/contrib/factorization/kernels:masked_matmul_ops",
        "//tensorflow/contrib/factorization/kernels:wals_solver_ops",
    ]),
)

tf_gen_op_libs([
    "factorization_ops",
])

cc_library(
    name = "all_ops",
    deps = [
        ":factorization_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_factorization_ops",
    out = "python/ops/gen_factorization_ops.py",
    deps = [
        ":factorization_ops_op_lib",
    ],
)

# Ops tests
tf_py_test(
    name = "gmm_test",
    size = "medium",
    srcs = [
        "python/ops/gmm_test.py",
    ],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
    ],
    shard_count = 4,
    tags = [
        "no_pip",  # b/38283730
        "notsan",  # Flaky: b/30756419
    ],
)

tf_py_test(
    name = "gmm_ops_test",
    size = "large",
    srcs = [
        "python/ops/gmm_ops_test.py",
    ],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variables",
    ],
    tags = ["notsan"],  # b/62863147
)

py_library(
    name = "factorization_ops_test_utils_py",
    srcs = [
        "python/ops/factorization_ops_test_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "factorization_ops_test",
    srcs = ["python/ops/factorization_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        ":factorization_ops_test_utils_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
    ],
    shard_count = 4,
)

# Estimators tests
py_test(
    name = "kmeans_test",
    size = "medium",
    srcs = ["python/ops/kmeans_test.py"],
    python_version = "PY2",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column:feature_column_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "wals_test",
    size = "medium",
    srcs = ["python/ops/wals_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        ":factorization_ops_test_utils_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
    shard_count = 4,
    tags = [
        "noasan",  # times out b/63678675
        "nomsan",
    ],
)

# Kernel tests
tf_py_test(
    name = "wals_solver_ops_test",
    srcs = ["python/kernel_tests/wals_solver_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        ":gen_factorization_ops",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
    ],
)

tf_py_test(
    name = "clustering_ops_test",
    srcs = ["python/kernel_tests/clustering_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "masked_matmul_ops_test",
    srcs = ["python/kernel_tests/masked_matmul_ops_test.py"],
    additional_deps = [
        ":gen_factorization_ops",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
    ],
)

cuda_py_test(
    name = "masked_matmul_benchmark",
    srcs = ["python/kernel_tests/masked_matmul_benchmark.py"],
    additional_deps = [
        ":gen_factorization_ops",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
    ],
    main = "python/kernel_tests/masked_matmul_benchmark.py",
)
