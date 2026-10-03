#include "src/models/qwen36_35b_a3b/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <thread>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/mmq/q36_mmq.h"

namespace gufo::models::qwen36_35b_a3b::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

bool KQuant(core::GgmlType type) {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ6_K;
}

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Dequantizes K-quant rows of `t` (every expert's) into `host` on the
  /// host, in parallel.
  static void DequantHalf(const TensorRef& t, _Float16* host) {
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    const std::size_t row_bytes = t.RowBytes();
    const std::size_t total_rows = static_cast<std::size_t>(t.rows) * t.experts;
    const unsigned workers =
        std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < workers; ++w) {
      threads.emplace_back([&, w] {
        std::vector<float> row(t.cols);
        for (std::size_t r = w; r < total_rows; r += workers) {
          const auto* in = src + r * row_bytes;
          switch (t.type) {
            case core::GgmlType::kQ4_K:
              gufo::quant::DequantizeQ4_K(in, row.data(), t.cols);
              break;
            case core::GgmlType::kQ5_K:
              gufo::quant::DequantizeQ5_K(in, row.data(), t.cols);
              break;
            default:
              gufo::quant::DequantizeQ6_K(in, row.data(), t.cols);
              break;
          }
          for (std::size_t c = 0; c < t.cols; ++c)
            host[r * t.cols + c] = static_cast<_Float16>(row[c]);
        }
      });
    }
    for (auto& thread : threads)
      thread.join();
  }

  /// Uploads `count` F16 values as a tracked device allocation.
  void* UploadHalf(const _Float16* host, std::size_t count,
                   std::string_view name) {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, count * sizeof(_Float16) + kTailMargin) != hipSuccess ||
        hipMemcpy(ptr, host, count * sizeof(_Float16), hipMemcpyHostToDevice) !=
            hipSuccess) {
      Fail("F16 copy failed for " + std::string(name));
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += count * sizeof(_Float16) + kTailMargin;
    return ptr;
  }

  /// F16 rows of a dense K-quant matrix.
  void* HalfRows(const TensorRef& t) {
    std::vector<_Float16> host(static_cast<std::size_t>(t.rows) * t.cols);
    DequantHalf(t, host.data());
    max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    return UploadHalf(host.data(), host.size(), t.name);
  }

  /// Q8_0 copy of F16 rows on the device; the F16 rows are released.
  void* PrefillQ8(void* half, std::size_t count) {
    if (half == nullptr || !ok)
      return nullptr;
    const std::size_t q8_bytes = count / 32 * 34;
    void* q8 = nullptr;
    if (hipMalloc(&q8, q8_bytes + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for a prefill Q8_0 copy");
      return nullptr;
    }
    allocations.push_back(q8);
    bytes += q8_bytes + kTailMargin;
    QuantizeRowsQ8_0(static_cast<const __half*>(half), q8, count, nullptr);
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(q8) + q8_bytes, 0, kTailMargin) !=
            hipSuccess) {
      Fail("prefill Q8_0 quantization failed");
      return nullptr;
    }
    std::erase(allocations, half);
    (void)hipFree(half);
    bytes -= count * sizeof(_Float16) + kTailMargin;
    return q8;
  }

  /// A K-quant table read one row at a time, kept only as F16 rows.
  DeviceTensor HalfTable(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok)
      return d;
    d.data = HalfRows(t);
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    return d;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    // Prefill GEMMs read F16 copies; the vocabulary head only ever projects
    // a few rows and keeps its compact encoding alone.
    if (d.kquant() && d.experts == 1 && d.rows <= 65536) {
      d.prefill =
          PrefillQ8(HalfRows(t), static_cast<std::size_t>(d.rows) * d.cols);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
    }
    // Routed Q6_K experts (a few down projections) have no routed WMMA
    // decoder; their Q8_0 copy keeps those layers on the WMMA route.
    if (t.type == core::GgmlType::kQ6_K && d.experts > 1) {
      const std::size_t count =
          static_cast<std::size_t>(d.rows) * d.experts * d.cols;
      std::vector<_Float16> host(count);
      DequantHalf(t, host.data());
      d.prefill = PrefillQ8(UploadHalf(host.data(), count, t.name), count);
    }
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 or BF16 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kBF16 &&
           type != core::GgmlType::kQ8_0 && !KQuant(type))) {
        Fail(
            "stacked upload needs F32, BF16, Q8_0 or K-quant tensors of one "
            "shape");
        return d;
      }
      rows += t->rows;
      // 16-bit inputs are widened to F32 before narrowing to F16.
      size += type == core::GgmlType::kQ8_0 || KQuant(type)
                  ? t->SizeBytes()
                  : t->ElementCount() * sizeof(float);
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      auto* dst = static_cast<std::uint8_t*>(ptr) + offset;
      if (type == core::GgmlType::kBF16) {
        // BF16 widens exactly to F32 on the host; the narrowing below then
        // treats it like the F32 routers.
        std::vector<float> wide(t->ElementCount());
        const auto* src = static_cast<const std::uint16_t*>(t->data);
        for (std::size_t i = 0; i < wide.size(); ++i) {
          const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
          std::memcpy(&wide[i], &bits, sizeof(float));
        }
        if (hipMemcpy(dst, wide.data(), wide.size() * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += wide.size() * sizeof(float);
        continue;
      }
      if (!stager.Copy(t->shard, t->file_offset, t->SizeBytes(), dst, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0 || KQuant(type)) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      if (type == core::GgmlType::kQ8_0) {
        max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
        return d;
      }
      // The stacked prefill copy keeps the same row order.
      std::vector<_Float16> host(static_cast<std::size_t>(rows) * d.cols);
      std::size_t row0 = 0;
      for (const TensorRef* t : parts) {
        DequantHalf(*t, host.data() + row0 * d.cols);
        row0 += t->rows;
      }
      d.prefill = PrefillQ8(
          UploadHalf(host.data(), host.size(), (*parts.begin())->name),
          host.size());
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.attn_norm = Copy(l.attn_norm);
    d.post_norm = Copy(l.post_norm);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != (*parts.begin())->type ||
            (t->type != core::GgmlType::kQ8_0 && !KQuant(t->type)) ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_shared_norm = Copy(l.nextn_shared_norm);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 const MtpWeights* mtp,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg};
  // The embedding lookup reads Q8_0 or 16-bit rows; a K-quant table is kept
  // as F16. The output head keeps its own encoding.
  const bool kquant_embd = w.token_embd.type == core::GgmlType::kQ4_K ||
                           w.token_embd.type == core::GgmlType::kQ5_K ||
                           w.token_embd.type == core::GgmlType::kQ6_K;
  m->token_embd_ =
      kquant_embd ? up.HalfTable(w.token_embd) : up.Copy(w.token_embd);
  m->output_ = w.output.data == w.token_embd.data && !kquant_embd
                   ? m->token_embd_
                   : up.Copy(w.output);
  m->output_norm_ = up.Copy(w.output_norm);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  // Dense Q6_K matrices are read only by DenseQ6KVec, in its row layout.
  {
    std::vector<DeviceTensor*> dense{&m->output_};
    std::vector<DeviceLayer*> layers;
    for (auto& l : m->layers_)
      layers.push_back(&l);
    if (m->has_mtp_)
      layers.push_back(&m->mtp_);
    for (DeviceLayer* l : layers) {
      for (DeviceTensor* t :
           {&l->ssm_in, &l->ssm_qkv, &l->ssm_gate, &l->ssm_out, &l->attn_qkv,
            &l->attn_q, &l->attn_k, &l->attn_v, &l->attn_out, &l->shexp_gate,
            &l->shexp_up, &l->shexp_down, &l->nextn_fc_embedding,
            &l->nextn_fc_hidden}) {
        dense.push_back(t);
      }
    }
    std::vector<const void*> done;
    for (DeviceTensor* t : dense) {
      if (t->type != core::GgmlType::kQ6_K || t->experts != 1 ||
          std::find(done.begin(), done.end(), t->data) != done.end()) {
        continue;
      }
      if (!RepackQ6KRows(t->data, t->rows, t->cols)) {
        if (error_msg != nullptr)
          *error_msg = "Q6_K repack failed";
        return nullptr;
      }
      done.push_back(t->data);
    }
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  // Prefill reads Q8_0 copies through the tuned Q8_0 kernels.
  m->prefill_layers_ = m->layers_;
  for (auto& l : m->prefill_layers_) {
    for (DeviceTensor* t :
         {&l.ssm_in, &l.ssm_qkv, &l.ssm_gate, &l.ssm_out, &l.attn_qkv,
          &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out, &l.shexp_gate,
          &l.shexp_up, &l.shexp_down, &l.ffn_down_exps}) {
      if (t->prefill != nullptr) {
        t->data = t->prefill;
        t->type = core::GgmlType::kQ8_0;
        t->prefill = nullptr;
      }
    }
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen36_35b_a3b::rocm
