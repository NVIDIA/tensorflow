# Description:
#   Contains ops for image manipulation.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core/platform:default/build_config_root.bzl", "if_dynamic_pywrap")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_custom_op_library(
    name = "python/ops/_image_ops.so",
    srcs = if_dynamic_pywrap([], otherwise = [
        "kernels/bipartite_match_op.cc",
        "kernels/image_ops.cc",
        "kernels/image_ops.h",
        "kernels/segmentation_ops.cc",
        "kernels/segmentation_ops.h",
        "ops/image_ops.cc",
    ]),
    gpu_srcs = if_dynamic_pywrap([], otherwise = [
        "kernels/image_ops_gpu.cu.cc",
        "kernels/image_ops.h",
    ]),
)

tf_kernel_library(
    name = "image_ops_kernels",
    srcs = [
        "kernels/bipartite_match_op.cc",
        "kernels/image_ops.cc",
        "kernels/image_ops.h",
        "kernels/segmentation_ops.cc",
        "kernels/segmentation_ops.h",
    ],
    gpu_srcs = [
        "kernels/image_ops_gpu.cu.cc",
        "kernels/image_ops.h",
    ],
    deps = [
        ":image_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["image_ops"],
)

tf_gen_op_wrapper_py(
    name = "image_ops",
    deps = [":image_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "image_py",
    srcs = [
        "__init__.py",
        "python/ops/image_ops.py",
    ],
    dso = [":python/ops/_image_ops.so"],
    kernels = [
        ":image_ops_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dense_image_warp_py",
        ":image_ops",
        ":interpolate_spline_py",
        ":sparse_image_warp_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:common_shapes",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/image_ops_test.py"],
    additional_deps = [
        ":distort_image_py",
        ":image_py",
        ":single_image_random_dot_stereograms_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "notap",  # b/136286905
    ],
)

tf_custom_op_library(
    name = "python/ops/_distort_image_ops.so",
    srcs = [
        "kernels/adjust_hsv_in_yiq_op.cc",
        "kernels/adjust_hsv_in_yiq_op.h",
        "ops/distort_image_ops.cc",
    ],
    gpu_srcs = [
        "kernels/adjust_hsv_in_yiq_op_gpu.cu.cc",
        "kernels/adjust_hsv_in_yiq_op.h",
    ],
    deps = [
        "//tensorflow/core/kernels:gpu_util_hdrs",
    ],
)

tf_kernel_library(
    name = "distort_image_ops_kernels",
    srcs = [
        "kernels/adjust_hsv_in_yiq_op.cc",
        "kernels/adjust_hsv_in_yiq_op.h",
    ],
    gpu_srcs = [
        "kernels/adjust_hsv_in_yiq_op_gpu.cu.cc",
        "kernels/adjust_hsv_in_yiq_op.h",
    ],
    deps = [
        ":distort_image_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:gpu_util_hdrs",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "adjust_hsv_in_yiq_op_test",
    size = "small",
    srcs = [
        "kernels/adjust_hsv_in_yiq_op.h",
        "kernels/adjust_hsv_in_yiq_op_test.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["distort_image_ops"],
)

tf_gen_op_wrapper_py(
    name = "distort_image_ops",
    deps = [":distort_image_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "distort_image_py",
    srcs = [
        "__init__.py",
        "python/ops/distort_image_ops.py",
    ],
    dso = [":python/ops/_distort_image_ops.so"],
    kernels = [
        ":distort_image_ops_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distort_image_ops",
        ":single_image_random_dot_stereograms_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "distort_image_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/distort_image_ops_test.py"],
    additional_deps = [
        ":distort_image_py",
        ":image_py",
        ":single_image_random_dot_stereograms_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "dense_image_warp_py",
    srcs = [
        "python/ops/dense_image_warp.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "interpolate_spline_py",
    srcs = [
        "python/ops/interpolate_spline.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "sparse_image_warp_py",
    srcs = [
        "python/ops/sparse_image_warp.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dense_image_warp_py",
        ":interpolate_spline_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "sparse_image_warp_test",
    size = "medium",
    srcs = ["python/kernel_tests/sparse_image_warp_test.py"],
    additional_deps = [
        ":sparse_image_warp_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:variables",
        "//tensorflow/core:protos_all_py",
    ],
    data = [":sparse_image_warp_test_data"],
    tags = ["no_pip"],
)

filegroup(
    name = "sparse_image_warp_test_data",
    srcs = glob(["python/kernel_tests/test_data/*.png"]),
)

cuda_py_test(
    name = "dense_image_warp_test",
    size = "medium",
    srcs = ["python/kernel_tests/dense_image_warp_test.py"],
    additional_deps = [
        ":dense_image_warp_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:variables",
        "//tensorflow/core:protos_all_py",
    ],
)

cuda_py_test(
    name = "interpolate_spline_test",
    size = "medium",
    srcs = ["python/kernel_tests/interpolate_spline_test.py"],
    additional_deps = [
        ":interpolate_spline_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:variables",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "segmentation_test",
    size = "medium",
    srcs = ["python/kernel_tests/segmentation_test.py"],
    additional_deps = [
        ":distort_image_py",
        ":image_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_custom_op_library(
    name = "python/ops/_single_image_random_dot_stereograms.so",
    srcs = [
        "kernels/single_image_random_dot_stereograms_ops.cc",
        "ops/single_image_random_dot_stereograms_ops.cc",
    ],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["single_image_random_dot_stereograms_ops"],
)

tf_kernel_library(
    name = "single_image_random_dot_stereograms_kernels",
    srcs = [
        "kernels/single_image_random_dot_stereograms_ops.cc",
    ],
    deps = [
        ":single_image_random_dot_stereograms_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_gen_op_wrapper_py(
    name = "single_image_random_dot_stereograms_ops",
    deps = [":single_image_random_dot_stereograms_ops_op_lib"],
)

alias(
    name = "image_ops_cc",
    actual = ":image_ops_op_lib",
)

tf_custom_op_py_library(
    name = "single_image_random_dot_stereograms_py",
    srcs = glob(["python/ops/single*.py"]) + ["__init__.py"],
    dso = [":python/ops/_single_image_random_dot_stereograms.so"],
    kernels = [
        ":single_image_random_dot_stereograms_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":image_py",
        ":single_image_random_dot_stereograms_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "single_image_random_dot_stereograms_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/single_image_random_dot_stereograms_ops_test.py"],
    additional_deps = [
        ":distort_image_py",
        ":image_py",
        ":single_image_random_dot_stereograms_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)
