#include <cassert>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "src/core/platform/publish.hpp"

namespace {
#if defined(__linux__)
bool fail_directory_flush = false;
unsigned directory_flushes = 0;
#endif

std::string Read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}

void Publish(const std::filesystem::path& path, std::string_view bytes) {
  std::string error;
  auto file = gufo::platform::AtomicPublishedFile::Create(path, &error);
  assert(file.valid());
  assert(file.Write(bytes.data(), bytes.size(), &error));
  assert(file.Commit(path, &error));
  assert(Read(path) == bytes);
}

#if !defined(_WIN32)
void DescriptorZero(const std::filesystem::path& root) {
  const int saved_stdin = ::dup(STDIN_FILENO);
  ::close(STDIN_FILENO);
  const auto lock_path = (root / "lock").string();
  {
    auto lock = gufo::platform::FileLock::Acquire(lock_path);
    auto moved = std::move(lock);
    lock = std::move(moved);
  }
  errno = 0;
  assert(::fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF);
  const int fd = ::open(lock_path.c_str(), O_RDWR);
  assert(fd == STDIN_FILENO);
  assert(::flock(fd, LOCK_EX | LOCK_NB) == 0);
  ::close(fd);

  const auto path = root / "descriptor-zero.bin";
  Publish(path, "fd zero");
  assert(::fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF);
  {
    std::string error;
    auto file = gufo::platform::AtomicPublishedFile::Create(path, &error);
    assert(file.valid());
    auto moved = std::move(file);
    file = std::move(moved);
    assert(file.valid() && !moved.valid());
  }
  assert(::fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF);
  if (saved_stdin >= 0) {
    assert(::dup2(saved_stdin, STDIN_FILENO) == STDIN_FILENO);
    ::close(saved_stdin);
  }
}
#endif
}  // namespace

#if defined(__linux__)
extern "C" int __real_fsync(int fd);
extern "C" int __wrap_fsync(int fd) {
  struct stat status{};
  if (::fstat(fd, &status) == 0 && S_ISDIR(status.st_mode)) {
    ++directory_flushes;
    if (fail_directory_flush) {
      errno = EIO;
      return -1;
    }
  }
  return __real_fsync(fd);
}
#endif

int main() {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("gufo-platform-publish-" +
       std::to_string(std::filesystem::file_time_type::clock::now()
                          .time_since_epoch()
                          .count()));
  std::filesystem::create_directories(root);
  Publish(root / "plans.bin", "first");
  Publish(root / "plans.bin", "replacement");
#if !defined(_WIN32)
  DescriptorZero(root);
#endif
#if defined(__linux__)
  const auto previous_directory = std::filesystem::current_path();
  std::filesystem::current_path(root);
  const unsigned before = directory_flushes;
  Publish("relative.bin", "relative");
  assert(directory_flushes == before + 1);
  std::string error;
  auto file =
      gufo::platform::AtomicPublishedFile::Create("relative.bin", &error);
  assert(file.valid() && file.Write("updated", 7, &error));
  fail_directory_flush = true;
  assert(!file.Commit("relative.bin", &error));
  fail_directory_flush = false;
  assert(error.find("file replaced") != std::string::npos);
  assert(error.find("flush containing directory") != std::string::npos);
  assert(Read("relative.bin") == "updated");
  std::filesystem::current_path(previous_directory);
#endif
  for (const auto& entry : std::filesystem::directory_iterator(root))
    assert(entry.path().filename().string().find(".tmp.") == std::string::npos);
  std::filesystem::remove_all(root);
}
