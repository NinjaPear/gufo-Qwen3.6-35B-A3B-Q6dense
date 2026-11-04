# Qwen3.6-35B-A3B (qwen35moe) model extension. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
# Compile options match the pre-extension CMakeLists.txt so the kernels and
# engine build exactly as before.

gufo_extension_target(
  NAME gufo_qwen36_35b_a3b
  SOURCES
    config.cpp
    weights.cpp
    probe.cpp
    continuation_identity.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_text
    Threads::Threads
)

if(TARGET gufo_qwen36_35b_a3b)
  target_include_directories(gufo_qwen36_35b_a3b PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

add_library(gufo_qwen36_35b_a3b_reference STATIC EXCLUDE_FROM_ALL
  ${GUFO_EXTENSION_BUILD_ROOT}/cpu_ops.cpp
  ${GUFO_EXTENSION_BUILD_ROOT}/reference.cpp)
target_include_directories(gufo_qwen36_35b_a3b_reference PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${GUFO_EXTENSION_BUILD_ROOT}
)
target_link_libraries(gufo_qwen36_35b_a3b_reference PUBLIC
  gufo_qwen36_35b_a3b gufo_core)
# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(
  ${GUFO_EXTENSION_BUILD_ROOT}/cpu_ops.cpp
  ${GUFO_EXTENSION_BUILD_ROOT}/reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(Q36_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(Q36_MMQ_DIR "${Q36_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # q36_mmq namespace. The adapter retains its C++17 dialect.
  set(Q36_MMQ_SOURCES
    ${Q36_MMQ_DIR}/q36_ggml_stubs.hip.cpp
    ${Q36_MMQ_DIR}/q36_mmq.hip.cpp
    ${Q36_MMQ_DIR}/mmid.hip.cpp
    ${Q36_MMQ_DIR}/mmvq.hip.cpp
    ${Q36_MMQ_DIR}/moe_wave64.hip.cpp
    ${Q36_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen36_35b_a3b_mmq STATIC ${Q36_MMQ_SOURCES})
  set_source_files_properties(${Q36_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${Q36_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_qwen36_35b_a3b_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen36_35b_a3b_mmq PRIVATE
    GGML_HIP_NO_VMM)
  target_compile_options(gufo_qwen36_35b_a3b_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen36_35b_a3b_mmq SYSTEM PUBLIC ${Q36_MMQ_DIR})
  target_link_libraries(gufo_qwen36_35b_a3b_mmq PUBLIC hip::host roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_qwen36_35b_a3b_hip
  SOURCES
    kernels/rocm/blaslt.cpp
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp
    kernels/rocm/batch.cpp
    engine.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen36_35b_a3b
    gufo_qwen_vision
    gufo_qwen36_35b_a3b_mmq
    gufo_compare
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_qwen36_35b_a3b_hip)
  gufo_extension_hip_sources(gufo_qwen36_35b_a3b_hip
    SOURCES
      kernels/rocm/kernels.hip.cpp
      kernels/rocm/w8a8_wave64.hip
  )
  set_source_files_properties(${Q36_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_options(gufo_qwen36_35b_a3b_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen36_35b_a3b_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(gufo_qwen36_35b_a3b_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
