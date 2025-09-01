
TRIBITS_SUBPACKAGE(Core)

ADD_SUBDIRECTORY(src)

# Add the targets for building the Thyra test library so that other packages
# can use the same test library without requiring Thyra tests be enabled.
ADD_SUBDIRECTORY(test/nonlinear/models)

TRIBITS_ADD_TEST_DIRECTORIES(test)

TRIBITS_ADD_TEST_DIRECTORIES(example)

TRIBITS_SUBPACKAGE_POSTPROCESS()

if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(thyracore PRIVATE --cuda-host-only)
endif()
