/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_NVTX_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_NVTX_UTILS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/nvtx3/nvToolsExt.h"

namespace tensorflow {

// Forward declarations.
class OpKernel;
class Tensor;

namespace nvtx {

namespace detail {

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
inline bool RangesEnabled() {
  static bool is_enabled = [] {
    bool _is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_ENABLE_NVTX_RANGES",
                                               /*default_val=*/false,
                                               &_is_enabled));
    return _is_enabled;
  }();
  return is_enabled;
}

// Note: The memory backing msg must persist until the result of this function
// has been consumed by an NVTX API.
void MakeAttributes(const char* msg, absl::string_view category,
                    nvtxEventAttributes_t* result);

}  // namespace detail

string GetThunkExecutionRangeMessage(absl::string_view cluster_name,
                                     absl::string_view op_name,
                                     absl::string_view op_type);

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }

  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  nvtxDomainHandle_t handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
};

template <class Subclass>
struct DomainSingleton {
  static nvtxDomainHandle_t GetSingleton() {
    static NvtxDomain domain(Subclass::kName);
    return domain;
  }
};

class NullDomain;

template <>
struct DomainSingleton<NullDomain> {
  static nvtxDomainHandle_t GetSingleton() { return nullptr; }
};

struct NullDomain : public DomainSingleton<NullDomain> {};

struct CoreDomain : public DomainSingleton<CoreDomain> {
  static constexpr const char* kName = "tensorflow-core";
};

struct MemAllocDomain : public DomainSingleton<MemAllocDomain> {
  static constexpr const char* kName = "tensorflow-memalloc";
};

// Creates an NVTX range corresponding to the scope of the object.
// Uses the NVTX push/pop APIs. Does nothing if !RangesEnabled().
// Domain must be a subclass of DomainSingleton.
template <class Domain = NullDomain>
class ScopedRangeIfEnabled {
 public:
  static nvtxDomainHandle_t domain() { return Domain::GetSingleton(); }

  ScopedRangeIfEnabled(absl::string_view category,
                       std::function<string()> msg_fn) {
    if (TF_PREDICT_TRUE(!detail::RangesEnabled())) return;
    string msg = msg_fn();
    nvtxEventAttributes_t attrs;
    detail::MakeAttributes(msg.c_str(), category, &attrs);
    if (domain()) {
      ::nvtxDomainRangePushEx(domain(), &attrs);
    } else {
      ::nvtxRangePushEx(&attrs);
    }
  }
  explicit ScopedRangeIfEnabled(std::function<string()> msg_fn)
      : ScopedRangeIfEnabled({}, msg_fn) {}
  ~ScopedRangeIfEnabled() {
    if (TF_PREDICT_TRUE(!detail::RangesEnabled())) return;
    if (domain()) {
      ::nvtxDomainRangePop(domain());
    } else {
      ::nvtxRangePop();
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedRangeIfEnabled);
};

// Creates an NVTX range corresponding to the lifetime of the object.
// Uses the NVTX start/end APIs. If !RangesEnabled(), then no range is created
// and the object remains in an empty state (operator bool() will return false).
// Domain must be a subclass of DomainSingleton.
template <class Domain = NullDomain>
class UniqueRange {
  // msg_fn must be a nullary callable returning string.
  template <class MsgFunc>
  UniqueRange(absl::string_view category, MsgFunc msg_fn) {
    reset(category, msg_fn);
  }

 public:
  static nvtxDomainHandle_t domain() { return Domain::GetSingleton(); }

  // Named constructor to make it clear that behavior is conditional on
  // RangesEnabled().
  // msg_fn must be a nullary callable returning string.
  template <class MsgFunc>
  static UniqueRange IfEnabled(absl::string_view category, MsgFunc msg_fn) {
    return UniqueRange(category, msg_fn);
  }
  // msg_fn must be a nullary callable returning string.
  template <class MsgFunc>
  static UniqueRange IfEnabled(MsgFunc msg_fn) {
    return IfEnabled({}, msg_fn);
  }
  UniqueRange() = default;
  ~UniqueRange() { reset(); }
  UniqueRange(UniqueRange&& tmp) : id_(tmp.id_) { tmp.id_ = 0; }
  UniqueRange& operator=(UniqueRange&& tmp) {
    reset();
    id_ = tmp.id_;
    tmp.id_ = 0;
    return *this;
  }

  void reset() {
    if (id_) {
      if (domain()) {
        ::nvtxDomainRangeEnd(domain(), id_);
      } else {
        ::nvtxRangeEnd(id_);
      }
      id_ = 0;
    }
  }

  // msg_fn must be a nullary callable returning string.
  template <class MsgFunc>
  void reset(absl::string_view category, MsgFunc msg_fn) {
    if (TF_PREDICT_TRUE(!detail::RangesEnabled())) return;
    reset();
    string msg = msg_fn();
    nvtxEventAttributes_t attrs;
    detail::MakeAttributes(msg.c_str(), category, &attrs);
    if (domain()) {
      id_ = ::nvtxDomainRangeStartEx(domain(), &attrs);
    } else {
      id_ = ::nvtxRangeStartEx(&attrs);
    }
  }
  // msg_fn must be a nullary callable returning string.
  template <class MsgFunc>
  void reset(MsgFunc msg_fn) {
    return reset({}, msg_fn);
  }

  nvtxRangeId_t get() const { return id_; }

  explicit operator bool() const { return static_cast<bool>(id_); }

 private:
  nvtxRangeId_t id_ = 0;
  TF_DISALLOW_COPY_AND_ASSIGN(UniqueRange);
};

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_NVTX_UTILS_H_
