# Description:
#   Contains ops to build an input pipeline for tensorflow.
#   APIs here are meant to evolve over time.

load("//tensorflow:tensorflow.bzl", "py_test")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_tests",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core/platform:default/build_config_root.bzl", "if_dynamic_pywrap")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku,ptucker): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_input_pipeline_ops.so",
    srcs = [
        "ops/input_pipeline_ops.cc",
    ],
    deps = if_dynamic_pywrap([], otherwise = [
        "//tensorflow/contrib/input_pipeline/kernels:input_pipeline_kernels",
    ]),
)

tf_gen_op_libs(
    op_lib_names = ["input_pipeline_ops"],
)

tf_gen_op_wrapper_py(
    name = "input_pipeline_ops",
    deps = [":input_pipeline_ops_op_lib"],
)

tf_kernel_library(
    name = "input_pipeline_ops_kernels",
    deps = [
        "//tensorflow/contrib/input_pipeline/kernels:input_pipeline_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "input_pipeline_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    dso = [":python/ops/_input_pipeline_ops.so"],
    kernels = [
        ":input_pipeline_ops_kernels",
        ":input_pipeline_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "input_pipeline_ops_test",
    size = "small",
    srcs = ["python/ops/input_pipeline_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_cc_tests(
    name = "input_pipeline_ops_test",
    size = "small",
    srcs = [
        "ops/input_pipeline_ops_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":input_pipeline_ops_op_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
