#ifndef GUFO_CORE_PLATFORM_SOCKET_HPP_
#define GUFO_CORE_PLATFORM_SOCKET_HPP_

/// Socket headers and the close-on-exec socket-type flag, so subsystem code
/// needs no per-OS includes. Include winsock2.h before windows.h on Windows,
/// with the macro guards that keep its min/max macros out of std:: calls.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

namespace gufo::platform {
// socket(2) on Windows has no close-on-exec type flag; created sockets stay
// inheritable.
inline constexpr int kSocketCloexec = 0;
}  // namespace gufo::platform
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

namespace gufo::platform {
inline constexpr int kSocketCloexec = SOCK_CLOEXEC;
}  // namespace gufo::platform
#endif

#endif  // GUFO_CORE_PLATFORM_SOCKET_HPP_
