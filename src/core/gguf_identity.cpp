#include "src/core/gguf_identity.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::core {
namespace {

class FileDescriptor {
public:
  explicit FileDescriptor(platform::PlatformFile file) : file_(file) {}
  ~FileDescriptor() { platform::CloseFile(file_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] platform::PlatformFile get() const { return file_; }

private:
  platform::PlatformFile file_;
};

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

std::string FileStamp(platform::PlatformFile file, std::size_t size) {
  platform::FileInfo info{};
  std::error_code ec;
  if (!platform::Stat(file, &info, ec) || !info.regular || info.size != size) {
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  }
  return std::to_string(info.device) + ':' + std::to_string(info.inode) + ':' +
         std::to_string(info.size) + ':' + std::to_string(info.modified_ns) +
         ':' + std::to_string(info.changed_ns);
}

platform::Directory OpenCacheDirectory() {
  auto base = platform::UserCacheDirectory();
  if (base.empty())
    return {};
  std::error_code error;
  base /= "gufo";
  std::filesystem::create_directories(base, error);
  if (error)
    return {};
  base /= "gguf-sha256-v1";
  (void)platform::MakeDirectory(base, error);
  platform::Directory directory;
  if (!directory.Open(base, error))
    return {};
  platform::FileInfo info{};
  if (!platform::StatAt(directory, ".", false, &info, error) ||
      !platform::OwnerOnly(info, 0077))
    return {};
  return directory;
}

std::string ReadDigest(const platform::Directory& directory,
                       const std::string& key) {
  if (!directory.valid())
    return {};
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, key, platform::FileMode::kRead, ec));
  platform::FileInfo info{};
  if (!platform::FileValid(file.get()) ||
      !platform::Stat(file.get(), &info, ec) || !info.regular ||
      !platform::OwnerOnly(info, 0022) || info.size != 65)
    return {};
  std::array<char, 65> bytes{};
  if (platform::Read(file.get(), bytes.data(), bytes.size(), ec) !=
          bytes.size() ||
      bytes.back() != '\n' ||
      !std::all_of(bytes.begin(), bytes.end() - 1, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return {};
  return {bytes.data(), bytes.size() - 1};
}

void StoreDigest(const platform::Directory& directory, const std::string& key,
                 const std::string& digest) {
  if (!directory.valid())
    return;
  static std::atomic<unsigned long> serial{0};
  const std::string temp = "." + key + "." +
                           std::to_string(platform::ProcessId()) + "." +
                           std::to_string(serial.fetch_add(1));
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, temp, platform::FileMode::kCreateNew, ec));
  if (!platform::FileValid(file.get()))
    return;
  const std::string contents = digest + '\n';
  if (platform::Write(file.get(), contents.data(), contents.size(), ec) ==
      contents.size()) {
    (void)platform::Rename(directory, temp, directory, key, ec);
  }
  (void)platform::Unlink(directory, temp, ec);
}

void HashFile(crypto::Sha256Hasher& hash, platform::PlatformFile file,
              std::size_t size) {
  constexpr std::size_t kChunk = 8 * 1024 * 1024;
  constexpr std::size_t kAlignment = 4096;
  constexpr std::size_t kReaders = 4;
  // Preserve the open inode and avoid filling the remaining UMA with a second
  // copy of weights after GPU loading. Hashing stays in exact file order.
  FileDescriptor direct(size >= kChunk ? platform::ReopenUnbuffered(file)
                                       : platform::kInvalidFile);
  struct AlignedDeleter {
    std::size_t alignment;
    void operator()(std::uint8_t* pointer) const noexcept {
      ::operator delete(pointer, std::align_val_t(alignment));
    }
  };
  struct Slot {
    std::unique_ptr<std::uint8_t, AlignedDeleter> buffer{
        nullptr, AlignedDeleter{kAlignment}};
    std::future<void> read;
    std::size_t size{0};
  };
  std::array<Slot, kReaders> slots;
  std::size_t next = 0;
  const auto queue = [&](Slot& slot) {
    if (next == size)
      return;
    if (!slot.buffer) {
      slot.buffer.reset(static_cast<std::uint8_t*>(
          ::operator new(kChunk, std::align_val_t(kAlignment))));
    }
    const auto offset = next;
    slot.size = std::min(kChunk, size - next);
    next += slot.size;
    slot.read = std::async(std::launch::async, [&, offset] {
      std::size_t done = 0;
      bool buffered = !platform::FileValid(direct.get());
      while (done < slot.size) {
        const auto remaining = slot.size - done;
        const auto requested =
            buffered ? remaining
                     : (remaining + kAlignment - 1) / kAlignment * kAlignment;
        std::error_code ec;
        const auto count =
            platform::ReadAt(buffered ? file : direct.get(), offset + done,
                             slot.buffer.get() + done, requested, ec);
        if (count == 0 && ec == std::errc::interrupted)
          continue;
        if (count == 0 && !buffered &&
            (ec == std::errc::invalid_argument ||
             ec == std::errc::operation_not_supported)) {
          buffered = true;
          continue;
        }
        if (count == 0)
          throw std::runtime_error("cannot read complete GGUF for identity");
        done += std::min(count, remaining);
        if (done % kAlignment != 0)
          buffered = true;
      }
    });
  };
  for (auto& slot : slots)
    queue(slot);
  for (std::size_t index = 0; slots[index].read.valid();
       index = (index + 1) % slots.size()) {
    auto& slot = slots[index];
    slot.read.get();
    hash.Update({slot.buffer.get(), slot.size});
    queue(slot);
  }
}

std::string RegionDigest(const GgufMappedRegion& region) {
  std::string stamp;
  if (platform::FileValid(region.file_descriptor))
    stamp = FileStamp(region.file_descriptor, region.size);
  platform::Directory cache =
      stamp.empty() ? platform::Directory{} : OpenCacheDirectory();
  crypto::Sha256Hasher key_hash;
  HashString(key_hash, stamp);
  const auto key = key_hash.FinishHex();
  std::string digest = ReadDigest(cache, key);
  const bool cached = !digest.empty();
  if (!cached) {
    crypto::Sha256Hasher hash;
    constexpr std::size_t chunk = 8 * 1024 * 1024;
    if (platform::FileValid(region.file_descriptor)) {
      HashFile(hash, region.file_descriptor, region.size);
    } else {
      const auto* bytes = static_cast<const std::uint8_t*>(region.data);
      for (std::size_t offset = 0; offset < region.size;) {
        const auto length = std::min(chunk, region.size - offset);
        hash.Update({bytes + offset, length});
        offset += length;
      }
    }
    digest = hash.FinishHex();
  }
  if (!stamp.empty() &&
      FileStamp(region.file_descriptor, region.size) != stamp) {
    throw std::runtime_error("GGUF file changed during identity lookup");
  }
  if (!cached)
    StoreDigest(cache, key, digest);
  return digest;
}

}  // namespace

std::string GgufIdentityHex(const GgufReader& reader) {
  std::vector<std::string> stamps;
  for (const auto& region : reader.GetMappedRegions())
    stamps.push_back(platform::FileValid(region.file_descriptor)
                         ? FileStamp(region.file_descriptor, region.size)
                         : std::string{});
  crypto::Sha256Hasher hash;
  HashString(hash, kGgufIdentityScheme);
  for (const auto& region : reader.GetMappedRegions()) {
    HashString(hash, ":" + std::to_string(region.size) + ":");
    HashString(hash, RegionDigest(region));
  }
  for (std::size_t index = 0; index < stamps.size(); ++index) {
    const auto& region = reader.GetMappedRegions()[index];
    if (!stamps[index].empty() &&
        stamps[index] != FileStamp(region.file_descriptor, region.size))
      throw std::runtime_error("GGUF artifact changed during identity lookup");
  }
  return hash.FinishHex();
}

}  // namespace gufo::core
