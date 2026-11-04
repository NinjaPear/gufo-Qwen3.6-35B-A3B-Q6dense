#ifndef GUFO_CORE_PLATFORM_PAGE_PIN_HPP_
#define GUFO_CORE_PLATFORM_PAGE_PIN_HPP_

#include <cstddef>

namespace gufo::platform {

/// Pins host pages for DMA, reducing request-side page faults. Returns a
/// non-negative pin token, or -1 when pinning is unavailable. The pin owns
/// no memory; callers keep the allocation alive until UnpinHostPages().
int PinHostPages(void* data, std::size_t bytes) noexcept;

/// Releases a pin from PinHostPages(). A negative token is ignored.
void UnpinHostPages(int pin) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PAGE_PIN_HPP_
