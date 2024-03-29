# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library_additional_deps",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_gpu_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
# Placeholder for Google-internal load statements.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt_stub",
    srcs = if_tensorrt([
        "stub/nvinfer_stub.cc",
        "stub/nvinfer_plugin_stub.cc",
    ]),
    textual_hdrs = glob(["stub/*.inc", "common/utils.h"]),
    deps = if_tensorrt([
        "@local_config_tensorrt//:tensorrt_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
    linkopts = if_tensorrt([
        "-Wl,-rpath,$$ORIGIN/../../tensorrt",
        "-Wl,-rpath,$$ORIGIN/../tensorrt",
    ]),
)

alias(
    name = "tensorrt_lib",
    actual = select({
        "//tensorflow:oss": ":tensorrt_stub",
        "//conditions:default": "@local_config_tensorrt//:tensorrt",
    }),
)

tf_cuda_cc_test(
    name = "tensorrt_test_cc",
    size = "small",
    srcs = ["tensorrt_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_logging",
        ":utils",
        "//tensorflow/core:gpu_init",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "trt_op_kernels",
    srcs = [
        "kernels/get_calibration_data_op.cc",
        "kernels/trt_engine_op.cc",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":trt_allocator",
        ":trt_conversion",
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        ":utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/core:core_cpu_lib_no_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/stream_executor/lib",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

cc_library(
    name = "trt_engine_resource_op_kernels",
    srcs = ["kernels/trt_engine_resource_ops.cc"],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":trt_allocator",
        ":trt_engine_instance_proto_cc",
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "trt_engine_resource_ops_test",
    size = "small",
    srcs = ["kernels/trt_engine_resource_ops_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_engine_instance_proto_cc",
        ":trt_engine_resource_op_kernels",
        ":trt_engine_resource_ops_op_lib",
        ":trt_logging",
        ":trt_resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:resource_variable_ops",
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "trt_engine_op_test",
    size = "small",
    srcs = ["kernels/trt_engine_op_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_op_kernels",
        ":trt_op_libs",
        ":trt_resources",
        ":trt_conversion",
        ":utils",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

tf_gen_op_libs(
    op_lib_names = [
        "trt_engine_op",
        "get_calibration_data_op",
        "trt_engine_resource_ops",
    ],
)

cc_library(
    name = "trt_op_libs",
    deps = [
        ":get_calibration_data_op_op_lib",
        ":trt_engine_op_op_lib",
    ],
)

tf_cuda_library(
    name = "trt_logging",
    srcs = ["utils/trt_logger.cc"],
    hdrs = ["utils/trt_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_gen_op_wrapper_py(
    name = "trt_ops",
    deps = [
        ":trt_engine_resource_ops_op_lib",
        ":trt_op_libs",
    ],
)

tf_custom_op_py_library(
    name = "trt_ops_loader",
    srcs_version = "PY2AND3",
    deps = [
        ":trt_ops",
        ":wrap_py_utils",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
    ],
)

tf_cuda_library(
    name = "trt_resources",
    srcs = [
        "utils/trt_int8_calibrator.cc",
        "utils/trt_lru_cache.cc",
    ],
    hdrs = [
        "utils/trt_int8_calibrator.h",
        "utils/trt_lru_cache.h",
        "utils/trt_tensor_proxy.h",
    ],
    deps = [
        ":trt_allocator",
        ":trt_logging",
        ":utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core:graph",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_library(
    name = "trt_allocator",
    srcs = ["utils/trt_allocator.cc"],
    hdrs = ["utils/trt_allocator.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cc_test(
    name = "trt_allocator_test",
    size = "small",
    srcs = ["utils/trt_allocator_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_allocator",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "trt_lru_cache_test",
    size = "small",
    srcs = ["utils/trt_lru_cache_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_resources",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Library for the node-level conversion portion of TensorRT operation creation
tf_cuda_library(
    name = "trt_conversion",
    srcs = [
        "convert/convert_graph.cc",
        "convert/convert_nodes.cc",
        "convert/trt_optimization_pass.cc",
    ],
    hdrs = [
        "convert/convert_graph.h",
        "convert/convert_nodes.h",
        "convert/trt_optimization_pass.h",
    ],
    deps = [
        ":segment",
        ":trt_allocator",
        ":trt_plugins",
        ":trt_logging",
        ":trt_resources",
        ":utils",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "convert_graph_test",
    size = "medium",
    srcs = ["convert/convert_graph_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_op_kernels",
        ":trt_op_libs",
        ":trt_conversion",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_cc_test(
    name = "convert_nodes_test",
    size = "medium",
    srcs = ["convert/convert_nodes_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_logging",
        ":trt_conversion",
        ":trt_plugins",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

# Library for the segmenting portion of TensorRT operation creation
cc_library(
    name = "segment",
    srcs = ["segment/segment.cc"],
    hdrs = [
        "segment/segment.h",
        "segment/union_find.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

tf_cuda_cc_test(
    name = "segment_test",
    size = "small",
    srcs = ["segment/segment_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":segment",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

#tf_gpu_kernel_library(
#    name = "plugin_cast",
#    srcs = ["plugin/plugin_cast.cu.cc"],
#    deps = [
#        ":trt_plugins",
#        "//tensorflow/core:framework_lite",
#    ] + if_tensorrt([
#        "@local_config_cuda//cuda:cuda_headers",
#        "@local_config_tensorrt//:tensorrt",
#    ]),
#)

tf_cuda_library(
    name = "trt_plugins",
    srcs = ["plugin/trt_plugin.cc"],
    hdrs = ["plugin/trt_plugin.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

cc_library(
    name = "utils",
    srcs = ["convert/utils.cc"],
    hdrs = [
        "common/utils.h",
        "convert/utils.h",
        "utils/trt_tensor_proxy.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
    ],
)

tf_proto_library(
    name = "trt_engine_instance_proto",
    srcs = ["utils/trt_engine_instance.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

cc_library(
    name = "py_utils",
    srcs = ["utils/py_utils.cc"],
    hdrs = ["utils/py_utils.h"],
    copts = tf_copts(),
    deps = if_tensorrt([
        ":tensorrt_lib",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
)

tf_py_wrap_cc(
    name = "wrap_py_utils",
    srcs = ["utils/py_utils.i"],
    copts = tf_copts(),
    deps = [
        ":py_utils",
        "//third_party/python_runtime:headers",
    ],
)
