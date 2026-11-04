#include "src/core/hip/detail/hipblaslt_plan_database.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <limits>
#include <ranges>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "src/core/platform/publish.hpp"

namespace gufo::hip::detail {
namespace {

struct DatabasePublication {
  platform::FileLock lock;
  platform::AtomicPublishedFile temporary;
};

constexpr std::array<char, 8> kMagic = {'S', 'T', 'R', 'I',
                                        'X', 'L', 'T', '\0'};
constexpr std::uint32_t kMaxRecords = 4096;
constexpr std::uint32_t kMaxStringBytes = 16 * 1024;
constexpr std::uint32_t kMaxAlgorithmBlobBytes = 64;

template<typename Integer>
bool WriteInteger(std::ostream& output, Integer value) {
  static_assert(std::is_integral_v<Integer>);
  using Unsigned = std::make_unsigned_t<Integer>;
  const auto bits = static_cast<Unsigned>(value);
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    output.put(static_cast<char>((bits >> (byte * 8)) & 0xffU));
  }
  return output.good();
}

template<typename Integer>
bool ReadInteger(std::istream& input, Integer* value) {
  static_assert(std::is_integral_v<Integer>);
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned bits = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    const int next = input.get();
    if (next == std::char_traits<char>::eof()) {
      return false;
    }
    bits |= static_cast<Unsigned>(static_cast<unsigned char>(next))
            << (byte * 8);
  }
  *value = static_cast<Integer>(bits);
  return true;
}

bool WriteString(std::ostream& output, const std::string& value) {
  if (value.size() > kMaxStringBytes ||
      value.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  if (!WriteInteger(output, static_cast<std::uint32_t>(value.size()))) {
    return false;
  }
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  return output.good();
}

bool ReadString(std::istream& input, std::string* value) {
  std::uint32_t size = 0;
  if (!ReadInteger(input, &size) || size > kMaxStringBytes) {
    return false;
  }
  value->resize(size);
  input.read(value->data(), static_cast<std::streamsize>(size));
  return input.good();
}

bool WriteBytes(std::ostream& output, const std::vector<std::uint8_t>& value) {
  if (value.empty() || value.size() > kMaxAlgorithmBlobBytes) {
    return false;
  }
  if (!WriteInteger(output, static_cast<std::uint32_t>(value.size()))) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(value.data()),
               static_cast<std::streamsize>(value.size()));
  return output.good();
}

bool ReadBytes(std::istream& input, std::vector<std::uint8_t>* value) {
  std::uint32_t size = 0;
  if (!ReadInteger(input, &size) || size == 0 ||
      size > kMaxAlgorithmBlobBytes) {
    return false;
  }
  value->resize(size);
  input.read(reinterpret_cast<char*>(value->data()),
             static_cast<std::streamsize>(size));
  return input.good();
}

struct RecordKey {
  std::uint64_t batch_size;
  std::uint64_t m;
  std::uint64_t k;
  std::uint32_t data_type;

  [[nodiscard]] bool operator==(const RecordKey&) const = default;
};

struct RecordKeyHash {
  [[nodiscard]] std::size_t operator()(const RecordKey& key) const noexcept {
    std::size_t hash = std::hash<std::uint64_t>{}(key.batch_size);
    hash ^= std::hash<std::uint64_t>{}(key.m) + 0x9e3779b9U + (hash << 6) +
            (hash >> 2);
    hash ^= std::hash<std::uint64_t>{}(key.k) + 0x9e3779b9U + (hash << 6) +
            (hash >> 2);
    hash ^= std::hash<std::uint32_t>{}(key.data_type) + 0x9e3779b9U +
            (hash << 6) + (hash >> 2);
    return hash;
  }
};

bool IsValidRecord(const HipblasLtPlanRecord& record) {
  return record.batch_size > 0 && record.m > 0 && record.k > 0 &&
         record.data_type == HipblasLtPlanDataType::kBfloat16 &&
         record.algorithm_id >= 0 && !record.algorithm_blob.empty() &&
         record.algorithm_blob.size() <= kMaxAlgorithmBlobBytes &&
         record.workspace_bytes <= (1ULL << 30);
}

}  // namespace

HipblasLtPlanDatabaseLoadResult InspectHipblasLtPlanDatabase(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return {
        .status = HipblasLtPlanDatabaseLoadStatus::kNotFound,
        .database = {},
        .error = "plan database not found",
    };
  }

  std::array<char, kMagic.size()> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  HipblasLtPlanDatabase database;
  std::uint32_t record_count = 0;
  if (!input.good() || magic != kMagic ||
      !ReadInteger(input, &database.schema_version) ||
      !ReadString(input, &database.key.hardware_fingerprint) ||
      !ReadInteger(input, &database.key.hip_runtime_version) ||
      !ReadInteger(input, &database.key.hipblaslt_version) ||
      !ReadInteger(input, &record_count) || record_count > kMaxRecords) {
    return {
        .status = HipblasLtPlanDatabaseLoadStatus::kInvalid,
        .database = {},
        .error = "invalid plan database header",
    };
  }
  if (database.schema_version != kHipblasLtPlanDatabaseSchemaVersion) {
    return {
        .status = HipblasLtPlanDatabaseLoadStatus::kIncompatible,
        .database = {},
        .error = "unsupported plan database schema",
    };
  }
  database.records.reserve(record_count);
  std::unordered_set<RecordKey, RecordKeyHash> keys;
  for (std::uint32_t index = 0; index < record_count; ++index) {
    HipblasLtPlanRecord record;
    std::uint32_t data_type = 0;
    if (!ReadInteger(input, &record.batch_size) ||
        !ReadInteger(input, &record.m) || !ReadInteger(input, &record.k) ||
        !ReadInteger(input, &data_type) ||
        !ReadInteger(input, &record.algorithm_id) ||
        !ReadBytes(input, &record.algorithm_blob) ||
        !ReadInteger(input, &record.workspace_bytes) ||
        !ReadInteger(input, &record.median_nanoseconds) ||
        !ReadString(input, &record.solution_name) ||
        !ReadString(input, &record.kernel_name)) {
      return {
          .status = HipblasLtPlanDatabaseLoadStatus::kInvalid,
          .database = {},
          .error = "truncated plan database record",
      };
    }
    record.data_type = static_cast<HipblasLtPlanDataType>(data_type);
    const RecordKey key{record.batch_size, record.m, record.k, data_type};
    if (!IsValidRecord(record) || !keys.insert(key).second) {
      return {
          .status = HipblasLtPlanDatabaseLoadStatus::kInvalid,
          .database = {},
          .error = "invalid or duplicate plan database record",
      };
    }
    database.records.push_back(std::move(record));
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    return {
        .status = HipblasLtPlanDatabaseLoadStatus::kInvalid,
        .database = {},
        .error = "unexpected trailing plan database data",
    };
  }

  return {
      .status = HipblasLtPlanDatabaseLoadStatus::kLoaded,
      .database = std::move(database),
      .error = {},
  };
}

HipblasLtPlanDatabaseLoadResult LoadHipblasLtPlanDatabase(
    const std::filesystem::path& path,
    const HipblasLtPlanDatabaseKey& expected_key) {
  auto result = InspectHipblasLtPlanDatabase(path);
  if (result.status == HipblasLtPlanDatabaseLoadStatus::kLoaded &&
      !(result.database.key == expected_key)) {
    return {
        .status = HipblasLtPlanDatabaseLoadStatus::kIncompatible,
        .database = {},
        .error = "hardware or ROCm version mismatch",
    };
  }
  return result;
}

bool SaveHipblasLtPlanDatabase(const std::filesystem::path& path,
                               const HipblasLtPlanDatabase& database,
                               std::string* error) {
  if (database.schema_version != kHipblasLtPlanDatabaseSchemaVersion ||
      database.records.size() > kMaxRecords ||
      database.key.hardware_fingerprint.empty()) {
    if (error != nullptr) {
      *error = "invalid plan database";
    }
    return false;
  }

  std::unordered_set<RecordKey, RecordKeyHash> keys;
  for (const auto& record : database.records) {
    const RecordKey key{record.batch_size, record.m, record.k,
                        static_cast<std::uint32_t>(record.data_type)};
    if (!IsValidRecord(record) || !keys.insert(key).second) {
      if (error != nullptr) {
        *error = "invalid or duplicate plan database record";
      }
      return false;
    }
  }

  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) {
      if (error != nullptr) {
        *error = "failed to create plan database directory";
      }
      return false;
    }
  }

  DatabasePublication publication;
  const auto lock_path = path.string() + ".lock";
  try {
    publication.lock = platform::FileLock::Acquire(lock_path);
  } catch (const std::system_error&) {
    if (error)
      *error = "failed to lock plan database";
    return false;
  }
  // Concurrent tuners may have discovered different shapes. Merge under the
  // writer lock; readers continue to see the previous complete publication.
  auto merged = database;
  const auto previous = LoadHipblasLtPlanDatabase(path, database.key);
  if (previous.status == HipblasLtPlanDatabaseLoadStatus::kLoaded) {
    for (const auto& record : previous.database.records) {
      const RecordKey key{record.batch_size, record.m, record.k,
                          static_cast<std::uint32_t>(record.data_type)};
      if (merged.records.size() < kMaxRecords && keys.insert(key).second)
        merged.records.push_back(record);
    }
  }
  publication.temporary = platform::AtomicPublishedFile::Create(path, error);
  if (!publication.temporary.valid()) {
    return false;
  }
  std::ostringstream output(std::ios::binary | std::ios::out);

  output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  bool valid =
      output.good() && WriteInteger(output, merged.schema_version) &&
      WriteString(output, merged.key.hardware_fingerprint) &&
      WriteInteger(output, merged.key.hip_runtime_version) &&
      WriteInteger(output, merged.key.hipblaslt_version) &&
      WriteInteger(output, static_cast<std::uint32_t>(merged.records.size()));
  for (const auto& record : merged.records) {
    valid =
        valid && WriteInteger(output, record.batch_size) &&
        WriteInteger(output, record.m) && WriteInteger(output, record.k) &&
        WriteInteger(output, static_cast<std::uint32_t>(record.data_type)) &&
        WriteInteger(output, record.algorithm_id) &&
        WriteBytes(output, record.algorithm_blob) &&
        WriteInteger(output, record.workspace_bytes) &&
        WriteInteger(output, record.median_nanoseconds) &&
        WriteString(output, record.solution_name) &&
        WriteString(output, record.kernel_name);
  }
  if (!valid || !output.good()) {
    if (error)
      *error = "failed to serialize plan database";
    return false;
  }
  const auto bytes = std::move(output).str();
  if (!publication.temporary.Write(bytes.data(), bytes.size(), error)) {
    if (error != nullptr)
      *error = "failed to write plan database: " + *error;
    return false;
  }
  if (!publication.temporary.Commit(path, error)) {
    if (error != nullptr)
      *error = "failed to replace plan database: " + *error;
    return false;
  }
  return true;
}

const HipblasLtPlanRecord* FindHipblasLtPlanRecord(
    const HipblasLtPlanDatabase& database, std::size_t batch_size,
    std::size_t m, std::size_t k, HipblasLtPlanDataType data_type) {
  const auto found =
      std::ranges::find_if(database.records, [&](const auto& record) {
        return record.batch_size == batch_size && record.m == m &&
               record.k == k && record.data_type == data_type;
      });
  return found == database.records.end() ? nullptr : &*found;
}

}  // namespace gufo::hip::detail
