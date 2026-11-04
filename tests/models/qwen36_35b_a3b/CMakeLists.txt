# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen36_35b_a3b_tests)

function(add_qwen36_35b_a3b_test name library labels)
  set(target qwen36_35b_a3b_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen36_35b_a3b.${name} COMMAND ${target})
  set_tests_properties(qwen36_35b_a3b.${name} PROPERTIES
    LABELS "qwen36_35b_a3b;${labels}")
  add_dependencies(qwen36_35b_a3b_tests ${target})
endfunction()

add_qwen36_35b_a3b_test(config gufo_qwen36_35b_a3b "cpu")

add_executable(qwen36_35b_a3b_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen36_35b_a3b_reference_probe PRIVATE
  gufo_qwen36_35b_a3b_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  add_library(qwen36_35b_a3b_test_hip INTERFACE)
  target_link_libraries(qwen36_35b_a3b_test_hip INTERFACE
    gufo_qwen36_35b_a3b_hip gufo_qwen36_35b_a3b_mmq hip::host)
  target_compile_definitions(qwen36_35b_a3b_test_hip INTERFACE ENGINE_ENABLE_HIP=1)
  foreach(name IN ITEMS mtp_ops blaslt_ops gdn_ops attention_ops
      moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops)
    add_qwen36_35b_a3b_test(${name} qwen36_35b_a3b_test_hip "hip;gpu;gfx1151")
  endforeach()

  add_executable(qwen36_35b_a3b_session_test EXCLUDE_FROM_ALL session_test.cpp)
  target_link_libraries(qwen36_35b_a3b_session_test PRIVATE gufo_serving
    qwen36_35b_a3b_test_hip)
  add_executable(qwen36_35b_a3b_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp)
  target_link_libraries(qwen36_35b_a3b_snapshot_test PRIVATE
    qwen36_35b_a3b_test_hip)
  add_custom_target(qwen36_35b_a3b_model_tests
    DEPENDS qwen36_35b_a3b_session_test qwen36_35b_a3b_snapshot_test)

  add_executable(qwen36_35b_a3b_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp)
  target_link_libraries(qwen36_35b_a3b_gpu_probe PRIVATE qwen36_35b_a3b_test_hip)
endif()
