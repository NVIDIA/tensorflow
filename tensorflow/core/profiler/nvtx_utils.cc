/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/nvtx_utils.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace nvtx {
namespace detail {

namespace {

// A helper function to decide whether to enable setting color information in
// NVTX ranges. This reduces performance slightly, as it requires hashing the
// type string of each range.
inline bool RangesColorEnabled() {
  static bool is_enabled = [] {
    bool is_disabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DISABLE_NVTX_RANGES_COLOR",
                                               /*default_val=*/false,
                                               &is_disabled));
    return !is_disabled;
  }();
  return is_enabled;
}

inline size_t hash_bytes(const char* data, size_t size, size_t seed = 5381) {
  for (size_t i = 0; i < size; ++i) seed = seed * 33 + *data++;
  return seed;
}

inline uint32_t get_color(size_t hash) {
  static constexpr const uint32_t colors[] = {
      0x00aedb, 0xa200ff, 0xf47835, 0xd41243, 0x8ec127, 0xffb3ba, 0xffdfba,
      0xffffba, 0xbaffc9, 0xbae1ff, 0xbbcbdb, 0x9ebd9e, 0xdd855c, 0xf1e8ca,
      0x745151, 0x2e4045, 0x83adb5, 0xc7bbc9, 0x5e3c58, 0xbfb5b2, 0xff77aa,
      0xaaff77, 0x77aaff, 0xffffff, 0x000000, 0x57b85a, 0x57b88b, 0x57b5b8,
      0x5785b8, 0x5a57b8, 0x8b57b8, 0xb8575a};
  static constexpr const int ncolor = sizeof(colors) / sizeof(colors[0]);
  return colors[hash % ncolor];
}

}  // namespace

void MakeAttributes(const char* msg, absl::string_view category,
                    nvtxEventAttributes_t* result) {
  *result = {};
  result->version = NVTX_VERSION;
  result->size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  result->messageType = NVTX_MESSAGE_TYPE_ASCII;
  result->message.ascii = msg;
  if (detail::RangesColorEnabled() && !category.empty()) {
    size_t hash = detail::hash_bytes(category.data(), category.size());
    uint32_t color = detail::get_color(hash);
    uint32_t category = static_cast<uint32_t>(hash);
    result->colorType = NVTX_COLOR_ARGB;
    result->color = color;
    result->category = category;
  }
}

}  // namespace detail

string GetThunkExecutionRangeMessage(absl::string_view cluster_name,
                                     absl::string_view op_name,
                                     absl::string_view op_type) {
  cluster_name = cluster_name.substr(0, cluster_name.find("__XlaCompile"));
  return strings::StrCat(op_type, ": ", cluster_name, "_1/xla_run/", op_name);
}

}  // namespace nvtx
}  // namespace tensorflow
