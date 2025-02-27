
#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

#
# B) Define the header and source files (and directories)
#

SET(HEADERS "")
SET(SOURCES "")
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.h)
APPEND_GLOB(HEADERS ${DIR}/*.hpp)
APPEND_GLOB(SOURCES ${DIR}/*.cpp)

# Must glob the binary dir last to get all of the auto-generated headers
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_BINARY_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)
APPEND_SET(HEADERS ${DIR}/${PACKAGE_NAME}_config.h )


# Sanity Checking of Macros
ASSERT_DEFINED(Kokkos_ENABLE_OPENMP)
ASSERT_DEFINED(Kokkos_ENABLE_THREADS)
ASSERT_DEFINED(Kokkos_ENABLE_CUDA)
ASSERT_DEFINED(Kokkos_ENABLE_SERIAL)



# Default Node Processing
IF(DEFINED Tpetra_DefaultNode)
  MESSAGE(STATUS "User-set default Node: ${Tpetra_DefaultNode}")
ELSE()
  SET(Tpetra_DefaultNode "")
ENDIF()

IF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosSYCLWrapperNode")
  IF (NOT Kokkos_ENABLE_SYCL)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_SYCL is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosSYCLWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_SYCL)
    MESSAGE (FATAL_ERROR "Tpetra_INST_SYCL is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosSYCLWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_SYCLWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosHIPWrapperNode")
  IF (NOT Kokkos_ENABLE_HIP)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_HIP is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosHIPWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_HIP)
    MESSAGE (FATAL_ERROR "Tpetra_INST_HIP is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosHIPWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_HIPWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosCudaWrapperNode")
  IF (NOT Kokkos_ENABLE_CUDA)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_CUDA is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosCudaWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_CUDA)
    MESSAGE (FATAL_ERROR "Tpetra_INST_CUDA is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosCudaWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_CUDAWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosOpenMPWrapperNode")
  IF (NOT Kokkos_ENABLE_OPENMP)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_OPENMP is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosOpenMPWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_OPENMP)
    MESSAGE (FATAL_ERROR "Tpetra_INST_OPENMP is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosOpenMPWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_OPENMPWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosThreadsWrapperNode")
  IF (NOT Kokkos_ENABLE_THREADS)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_THREADS is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosThreadsWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_PTHREAD)
    MESSAGE (FATAL_ERROR "Tpetra_INST_PTHREAD is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosThreadsWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_THREADSWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosSerialWrapperNode")
  IF (NOT Kokkos_ENABLE_SERIAL)
    MESSAGE (FATAL_ERROR "Kokkos_ENABLE_SERIAL is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosSerialWrapperNode.")
  ENDIF ()
  IF (NOT Tpetra_INST_SERIAL)
    MESSAGE (FATAL_ERROR "Tpetra_INST_SERIAL is OFF, but you set Tpetra's default Node type to Kokkos::Compat::KokkosSerialWrapperNode.")
  ENDIF ()
  SET (HAVE_TPETRA_DEFAULTNODE_SERIALWRAPPERNODE ON CACHE BOOL "")
ELSEIF (Tpetra_DefaultNode STREQUAL "")

  #
  # The user didn't set Tpetra_DefaultNode, so we need to pick it.
  #
  IF (Tpetra_INST_SYCL)
    IF(NOT Kokkos_ENABLE_SYCL)
      MESSAGE(FATAL_ERROR "Tpetra_INST_SYCL is ON, but Kokkos_ENABLE_SYCL is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosSYCLWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosSYCLWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_SYCLWRAPPERNODE ON CACHE BOOL "")
  ELSEIF (Tpetra_INST_HIP)
    IF(NOT Kokkos_ENABLE_HIP)
      MESSAGE(FATAL_ERROR "Tpetra_INST_HIP is ON, but Kokkos_ENABLE_HIP is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosHIPWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosHIPWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_HIPWRAPPERNODE ON CACHE BOOL "")  
  ELSEIF (Tpetra_INST_CUDA)
    IF(NOT Kokkos_ENABLE_CUDA)
      MESSAGE(FATAL_ERROR "Tpetra_INST_CUDA is ON, but Kokkos_ENABLE_CUDA is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosCudaWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosCudaWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_CUDAWRAPPERNODE ON CACHE BOOL "")  
  ELSEIF (Tpetra_INST_OPENMP)
    IF(NOT Kokkos_ENABLE_OPENMP)
      MESSAGE(FATAL_ERROR "Tpetra_INST_OPENMP is ON, but Kokkos_ENABLE_OPENMP is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosOpenMPWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosOpenMPWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_OPENMPWRAPPERNODE ON CACHE BOOL "")  
  ELSEIF (Tpetra_INST_SERIAL)
    IF(NOT Kokkos_ENABLE_SERIAL)
      MESSAGE(FATAL_ERROR "Tpetra_INST_SERIAL is ON, but Kokkos_ENABLE_SERIAL is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosSerialWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosSerialWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_SERIALWRAPPERNODE ON CACHE BOOL "")  
  ELSEIF (Tpetra_INST_PTHREAD)
    IF(NOT Kokkos_ENABLE_THREADS)
      MESSAGE(FATAL_ERROR "Tpetra_INST_PTHREAD is ON, but Kokkos_ENABLE_THREADS is OFF.")
    ENDIF()
    MESSAGE(STATUS "Setting default Node to Kokkos::Compat::KokkosThreadsWrapperNode.")
    SET(Tpetra_DefaultNode "Kokkos::Compat::KokkosThreadsWrapperNode")
    SET(HAVE_TPETRA_DEFAULTNODE_THREADSWRAPPERNODE ON CACHE BOOL "")  
  ELSE ()
    # Extra check for the fix for Issue #190.
    MESSAGE(FATAL_ERROR "Tpetra cannot pick a default Node type if all Node types are disabled.")
  ENDIF ()
ELSE ()
  #
  # The user set Tpetra_DefaultNode to something other than 
  # the empty string, but we don't recognize their string.
  #
  MESSAGE(FATAL_ERROR "Unrecognized or disabled Node type \"${Tpetra_DefaultNode}\".  None of the possibilities Kokkos::Compat::KokkosSerialWrapperNode, Kokkos::Compat::KokkosOpenMPWrapperNode, Kokkos::Compat::KokkosThreadsWrapperNode, or Kokkos::Compat::KokkosCudaWrapperNode are enabled, so we do not know what default Node type to pick.")
ENDIF()
SET(Tpetra_DefaultNode "${Tpetra_DefaultNode}" CACHE STRING "" FORCE)
# mfh 04 Oct 2015: Set this for backwards compatibility ONLY.
SET(KokkosClassic_DefaultNode "${Tpetra_DefaultNode}" CACHE STRING "" FORCE)

CONFIGURE_FILE(cmake/KokkosClassic_DefaultNode_config.h.in  KokkosClassic_DefaultNode_config.h)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
APPEND_SET(HEADERS  ${CMAKE_CURRENT_BINARY_DIR}/KokkosClassic_DefaultNode_config.h)


TRIBITS_ADD_LIBRARY(
  tpetraclassic
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  ADDED_LIB_TARGET_NAME_OUT TPETRA_LIBNAME
  )

# We need to set the linker language explicitly here for CUDA builds.
SET_PROPERTY(
  TARGET ${TPETRA_LIBNAME}
  APPEND PROPERTY LINKER_LANGUAGE CXX
  )
