load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_protos_grappler",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["topological_sort.h"])

cc_library(
    name = "scc",
    srcs = ["scc.cc"],
    hdrs = ["scc.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:utils",
    ],
)

tf_cc_test(
    name = "scc_test",
    srcs = ["scc_test.cc"],
    data = [
        "//tensorflow/core/grappler/costs:graph_properties_testdata",
    ],
    deps = [
        ":scc",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "topological_sort",
    srcs = ["topological_sort.cc"],
    hdrs = ["topological_sort.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "topological_sort_test",
    srcs = ["topological_sort_test.cc"],
    deps = [
        ":topological_sort",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cc"],
    hdrs = ["frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "frame_test",
    size = "small",
    srcs = ["frame_test.cc"],
    deps = [
        ":frame",
        ":graph_view",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "traversal",
    srcs = ["traversal.cc"],
    hdrs = ["traversal.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "traversal_test",
    srcs = ["traversal_test.cc"],
    deps = [
        ":traversal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grappler_test",
    testonly = 1,
    srcs = [
        "grappler_test.cc",
    ],
    hdrs = ["grappler_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "grappler_test_test",
    size = "small",
    srcs = ["grappler_test_test.cc"],
    deps = [
        ":grappler_test",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_library(
    name = "functions",
    srcs = [
        "functions.cc",
    ],
    hdrs = ["functions.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_base_no_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "functions_test",
    srcs = ["functions_test.cc"],
    deps = [
        ":functions",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "colocation",
    srcs = ["colocation.cc"],
    hdrs = ["colocation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "colocation_test",
    size = "small",
    srcs = ["colocation_test.cc"],
    deps = [
        ":colocation",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "symbolic_shapes",
    srcs = ["symbolic_shapes.cc"],
    hdrs = ["symbolic_shapes.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "symbolic_shapes_test",
    srcs = ["symbolic_shapes_test.cc"],
    deps = [
        ":symbolic_shapes",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "tpu",
    srcs = ["tpu.cc"],
    hdrs = ["tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "tpu_test",
    srcs = ["tpu_test.cc"],
    deps = [
        ":tpu",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "canonicalizer",
    srcs = ["canonicalizer.cc"],
    hdrs = ["canonicalizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "canonicalizer_test",
    size = "small",
    srcs = ["canonicalizer_test.cc"],
    deps = [
        ":canonicalizer",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "graph_view_internal",
    hdrs = ["graph_view_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "graph_view_internal_test",
    srcs = ["graph_view_internal_test.cc"],
    deps = [
        ":graph_view",
        ":graph_view_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "graph_view_test",
    srcs = ["graph_view_test.cc"],
    deps = [
        ":graph_view",
        ":grappler_test",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)
