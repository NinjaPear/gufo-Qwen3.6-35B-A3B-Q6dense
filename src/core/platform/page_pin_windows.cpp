#include "src/core/platform/page_pin.hpp"

namespace gufo::platform {

// Windows exposes no io_uring-style register-pinned-buffer facility; HIP
// registration alone retains the allocation.
int PinHostPages(void*, std::size_t) noexcept {
  return -1;
}

void UnpinHostPages(int) noexcept {}

}  // namespace gufo::platform
