#ifndef GUFO_CORE_MAPPED_PREFETCH_HPP_
#define GUFO_CORE_MAPPED_PREFETCH_HPP_

#include <cstddef>

#include "src/core/platform/prefault.hpp"

namespace gufo::core {

/// Populate an existing readable mapping before HIP registration or upload.
/// Kept as a forwarding wrapper over platform::PrefaultMappedRange for the
/// model loaders; new code should call the platform function directly.
inline void PrefaultMappedRange(const void* data, std::size_t bytes) {
  platform::PrefaultMappedRange(data, bytes);
}

}  // namespace gufo::core

#endif  // GUFO_CORE_MAPPED_PREFETCH_HPP_
