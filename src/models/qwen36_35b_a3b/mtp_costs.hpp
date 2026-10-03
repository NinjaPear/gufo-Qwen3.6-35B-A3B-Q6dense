#ifndef GUFO_MODELS_QWEN36_35B_A3B_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen36_35b_a3b {

// gfx1151, Qwen3.6-35B-A3B Q6_K-dense with the in-GGUF Q8_0 MTP block,
// 2026-10-02. Milliseconds per cycle, including catch-up, recursive
// proposals and target verification. Rows are widths 1..8 (width 1 is
// headless MTP catch-up plus ordinary decode). C1 is measured with
// gpu_probe --bench --mtp at fixed widths (context 0 measured at 512);
// the batched cohorts keep Flash-Next's ratio to C1 until they are
// measured on this model.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {14.4F, 18.9F, 22.6F, 26.1F, 29.8F, 33.2F, 35.7F, 38.9F},      // C1
        {17.2F, 23.8F, 29.8F, 36.0F, 43.8F, 49.0F, 54.1F, 58.6F},      // C2
        {21.5F, 32.2F, 43.7F, 52.2F, 65.1F, 74.1F, 85.0F, 94.2F},      // C4
        {25.6F, 41.8F, 56.5F, 69.7F, 85.1F, 102.7F, 116.9F, 130.9F},   // C6
        {29.4F, 48.7F, 67.4F, 85.7F, 106.5F, 125.7F, 144.9F, 165.2F},  // C8
    },
    {
        // Context 4096
        {14.8F, 19.1F, 23.5F, 26.4F, 30.7F, 34.1F, 37.0F, 40.2F},      // C1
        {18.3F, 24.9F, 31.6F, 36.8F, 45.3F, 50.1F, 54.5F, 58.1F},      // C2
        {23.2F, 34.0F, 46.6F, 53.9F, 67.1F, 75.7F, 85.7F, 94.4F},      // C4
        {28.0F, 44.7F, 61.5F, 73.1F, 90.6F, 106.7F, 120.1F, 132.3F},   // C6
        {32.8F, 52.0F, 74.5F, 90.4F, 114.9F, 131.2F, 150.9F, 168.5F},  // C8
    },
    {
        // Context 32768
        {17.7F, 22.4F, 27.0F, 30.5F, 34.8F, 38.8F, 41.8F, 45.3F},       // C1
        {22.9F, 29.8F, 36.5F, 42.7F, 50.8F, 56.8F, 61.1F, 65.9F},       // C2
        {30.7F, 41.8F, 55.3F, 64.4F, 78.2F, 89.1F, 98.4F, 107.2F},      // C4
        {38.3F, 55.5F, 73.3F, 88.5F, 105.2F, 125.1F, 136.8F, 149.1F},   // C6
        {45.9F, 65.8F, 88.6F, 109.8F, 135.8F, 157.8F, 174.3F, 197.5F},  // C8
    },
};

// Sampled policy uses fixed configured capacity for reproducibility; greedy
// batch policy uses these curves to bootstrap each physical occupancy.
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = kMtpCycleMilliseconds[interval][cohort][i];
    const auto high = kMtpCycleMilliseconds[interval + 1][cohort][i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal =
          high - kMtpCycleMilliseconds[interval + 1][cohort][i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // GUFO_MODELS_QWEN36_35B_A3B_MTP_COSTS_HPP_
