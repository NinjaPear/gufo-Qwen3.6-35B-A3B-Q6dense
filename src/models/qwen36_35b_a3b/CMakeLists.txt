# Qwen3.6-35B-A3B (qwen35moe) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
add_library(gufo_qwen36_35b_a3b STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_qwen36_35b_a3b PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen36_35b_a3b PUBLIC gufo_core Threads::Threads)

add_library(gufo_qwen36_35b_a3b_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen36_35b_a3b_reference PUBLIC
  gufo_qwen36_35b_a3b)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(Q36_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
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

  set(Q36_HIP_SOURCES
    ${Q36_ROCM_DIR}/kernels.hip.cpp
    ${Q36_ROCM_DIR}/w8a8_wave64.hip
    ${Q36_ROCM_DIR}/blaslt.cpp
    ${Q36_ROCM_DIR}/device_model.cpp
    ${Q36_ROCM_DIR}/executor.cpp
    ${Q36_ROCM_DIR}/batch.cpp
    engine.cpp)
  target_sources(gufo_qwen36_35b_a3b PRIVATE ${Q36_HIP_SOURCES})
  set_source_files_properties(${Q36_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${Q36_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_options(gufo_qwen36_35b_a3b PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen36_35b_a3b SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen36_35b_a3b PUBLIC
    gufo_qwen36_35b_a3b_mmq hip::host roc::hipblas roc::hipblaslt)
endif()
