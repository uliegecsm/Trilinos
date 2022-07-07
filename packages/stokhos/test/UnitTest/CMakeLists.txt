ASSERT_DEFINED(PACKAGE_SOURCE_DIR CMAKE_CURRENT_SOURCE_DIR)

# Use this function to add a test "TEST_NAME" that will be added conditionally on
# the value of "${VARIABLE_PREFIX}_${SPACE_CHECK}".
# Note that if the execution space matches "Serial", "RUN_SERIAL" is not added to
# "TRIBITS_ADD_EXECUTABLE_AND_TEST".
# The test compile unit will have access to the execution space through the defined variable "STOKHOS_TEST_SPACE".
# By default, all the test is defined in:
#     Stokhos_${TEST_NAME}.hpp
# and the main entrypoint is in
#     Stokhos_${TEST_NAME}_main.cpp
# If your test needs a specific entrypoint, just create it under the file name
#     Stokhos_${TEST_NAME}_${SPACE}.cpp
# and it will be used instead of the default entrypoint using if/else clause.
function(STOKHOS_ADD_EXECUTABLE_AND_TEST)
    cmake_parse_arguments(
        PARSE
        ""
        "TEST_NAME;VARIABLE_PREFIX;NUM_MPI_PROCS"
        "SPACES_CHECK;SPACES_KOKKOS"
        "${ARGN}"
    )
    tribits_check_for_unparsed_arguments()
    list(LENGTH PARSE_SPACES_CHECK  length_space_check)
    list(LENGTH PARSE_SPACES_KOKKOS length_space_kokkos)
    foreach(space_check space_kokkos IN ZIP_LISTS PARSE_SPACES_CHECK PARSE_SPACES_KOKKOS)
        if(${${PARSE_VARIABLE_PREFIX}_${space_check}})
            if(NOT "${space_kokkos}" STREQUAL "Serial")
                set(RUN_SERIAL RUN_SERIAL)
            else()
                SET(RUN_SERIAL )
            endif()
            set(sources "Stokhos_${PARSE_TEST_NAME}.hpp")
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Stokhos_${PARSE_TEST_NAME}_${space_kokkos}.cpp")
                list(APPEND sources "Stokhos_${PARSE_TEST_NAME}_${space_kokkos}.cpp")
            elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Stokhos_${PARSE_TEST_NAME}_main.cpp")
                list(APPEND sources "Stokhos_${PARSE_TEST_NAME}_main.cpp")
            else()
                message(FATAL_ERROR "I don't know what to do for ${PARSE_TEST_NAME} with space ${space_kokkos}...")
            endif()
            TRIBITS_ADD_EXECUTABLE_AND_TEST(
                "${PARSE_TEST_NAME}_${space_kokkos}"
                SOURCES "${sources}"
                COMM serial mpi
                STANDARD_PASS_OUTPUT
                NUM_MPI_PROCS ${PARSE_NUM_MPI_PROCS}
                ${RUN_SERIAL}
            )
            target_compile_definitions(
                "Stokhos_${PARSE_TEST_NAME}_${space_kokkos}"
                PRIVATE
                    STOKHOS_TEST_SPACE=${space_kokkos}
            )
        endif()
    endforeach()
endfunction()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LegendreBasisUnitTest
  SOURCES Stokhos_LegendreBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  NormalizedLegendreBasisUnitTest
  SOURCES Stokhos_NormalizedLegendreBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  HermiteBasisUnitTest
  SOURCES Stokhos_HermiteBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  NormalizedHermiteBasisUnitTest
  SOURCES Stokhos_NormalizedHermiteBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  JacobiBasisUnitTest
  SOURCES Stokhos_JacobiBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  DerivExpansionUnitTest
#  SOURCES Stokhos_DerivExpansionUnitTest.cpp
#  COMM serial mpi
#  STANDARD_PASS_OUTPUT
#  NUM_MPI_PROCS 1
#  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  QuadExpansionUnitTest
  SOURCES Stokhos_QuadExpansionUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  QuadraturePseudoSpectralExpansionUnitTest
  SOURCES Stokhos_QuadraturePseudoSpectralExpansionUnitTest.cpp
          Stokhos_PseudoSpectralExpansionUnitTest.hpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TensorProductPseudoSpectralExpansionUnitTest
  SOURCES Stokhos_TensorProductPseudoSpectralExpansionUnitTest.cpp
          Stokhos_PseudoSpectralExpansionUnitTest.hpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SmolyakPseudoSpectralExpansionUnitTest
  SOURCES Stokhos_SmolyakPseudoSpectralExpansionUnitTest.cpp
          Stokhos_PseudoSpectralExpansionUnitTest.hpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  AlgebraicExpansionUnitTest
  SOURCES Stokhos_AlgebraicExpansionUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ConstantExpansionUnitTest
  SOURCES Stokhos_ConstantExpansionUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DivisionOperatorUnitTest
  SOURCES Stokhos_DivisionOperatorUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

IF(TPL_ENABLE_ForUQTK)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    ForUQTKExpansionUnitTest
    SOURCES Stokhos_ForUQTKExpansionUnitTest.cpp
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 1
    )

ENDIF()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  StieltjesUnitTest
  SOURCES Stokhos_StieltjesUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LanczosUnitTest
  SOURCES Stokhos_LanczosUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  GramSchmidtUnitTest
  SOURCES Stokhos_GramSchmidtUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Sparse3TensorUnitTest
  SOURCES Stokhos_Sparse3TensorUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ExponentialRandomFieldUnitTest
  SOURCES Stokhos_ExponentialRandomFieldUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LogNormalUnitTest
  SOURCES Stokhos_LogNormalUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SDMUtilsUnitTest
  SOURCES Stokhos_SDMUtilsUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ProductBasisUtilsUnitTest
  SOURCES Stokhos_ProductBasisUtilsUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TensorProductBasisUnitTest
  SOURCES Stokhos_TensorProductBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TotalOrderBasisUnitTest
  SOURCES Stokhos_TotalOrderBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SmolyakBasisUnitTest
  SOURCES Stokhos_SmolyakBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TensorProductPseudoSpectralOperatorUnitTest
  SOURCES Stokhos_TensorProductPseudoSpectralOperatorUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LexicographicTreeBasisUnitTest
  SOURCES Stokhos_LexicographicTreeBasisUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SparseGridQuadratureUnitTest
  SOURCES Stokhos_SparseGridQuadratureUnitTest.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  NUM_MPI_PROCS 1
  )

IF(Stokhos_ENABLE_TriKota)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    SmolyakPseudoSpectralOperatorUnitTest
    SOURCES Stokhos_SmolyakPseudoSpectralOperatorUnitTest.cpp
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 1
    )

ENDIF()

IF (Stokhos_ENABLE_EpetraExt)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    MatrixFreeOperatorUnitTest
    SOURCES Stokhos_MatrixFreeOperatorUnitTest.cpp
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 1
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    InterlacedOpUnitTest
    SOURCES Stokhos_InterlacedOpUnitTest.cpp
            Stokhos_InterlacedTestSupport.cpp
            Stokhos_InterlacedTestSupport.hpp
            ${TEUCHOS_STD_UNIT_TEST_MAIN}
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 2
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    BasisInteractionGraphUnitTest
    SOURCES Stokhos_BasisInteractionGraphUnitTest.cpp
            Stokhos_InterlacedTestSupport.cpp
            Stokhos_InterlacedTestSupport.hpp
            ${TEUCHOS_STD_UNIT_TEST_MAIN}
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 1
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AdaptivityToolsUnitTest
    SOURCES Stokhos_AdaptivityToolsUnitTest.cpp
            Stokhos_InterlacedTestSupport.cpp
            Stokhos_InterlacedTestSupport.hpp
            ${TEUCHOS_STD_UNIT_TEST_MAIN}
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 1
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    BuildColBasisUnitTest
    SOURCES Stokhos_BuildColBasisUnitTest.cpp
            Stokhos_InterlacedTestSupport.cpp
            Stokhos_InterlacedTestSupport.hpp
            ${TEUCHOS_STD_UNIT_TEST_MAIN}
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 2
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    InterlacedMapUnitTest
    SOURCES Stokhos_InterlacedMapUnitTest.cpp
            Stokhos_InterlacedTestSupport.cpp
            Stokhos_InterlacedTestSupport.hpp
            ${TEUCHOS_STD_UNIT_TEST_MAIN}
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 2
    )

ENDIF()

IF(Stokhos_ENABLE_Sacado)

  TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${PACKAGE_SOURCE_DIR}/../sacado/test/utils)

  SET(UTILS_SOURCES
    ${PACKAGE_SOURCE_DIR}/../sacado/test/utils/Sacado_Random.hpp
    ${PACKAGE_SOURCE_DIR}/../sacado/test/utils/Sacado_RandomImp.hpp
  )

  TRIBITS_COPY_FILES_TO_BINARY_DIR(copyMueLuFiles
    DEST_FILES muelu_cheby.xml
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    SacadoTraitsUnitTest
    SOURCES Stokhos_SacadoTraitsUnitTest.cpp
    ARGS
    COMM serial mpi
    NUM_MPI_PROCS 1
    STANDARD_PASS_OUTPUT
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    SacadoPromoteUnitTest
    SOURCES Stokhos_SacadoPromoteUnitTest.cpp
    ARGS
    COMM serial mpi
    NUM_MPI_PROCS 1
    STANDARD_PASS_OUTPUT
    )

  IF(Stokhos_ENABLE_PCE_Scalar_Type)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoPCEUnitTest
      SOURCES Stokhos_SacadoPCEUnitTest.cpp Stokhos_SacadoPCEUnitTest.hpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoETPCEUnitTest
      SOURCES Stokhos_SacadoETPCEUnitTest.cpp Stokhos_SacadoPCEUnitTest.hpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoPCESerializationTests
      SOURCES Stokhos_SacadoPCESerializationTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoPCECommTests
      SOURCES Stokhos_SacadoPCECommTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoUQPCEUnitTest
      SOURCES Stokhos_SacadoUQPCEUnitTest.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoUQPCESerializationTests
      SOURCES Stokhos_SacadoUQPCESerializationTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoUQPCECommTests
      SOURCES Stokhos_SacadoUQPCECommTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    STOKHOS_ADD_EXECUTABLE_AND_TEST(
        TEST_NAME       KokkosViewUQPCEUnitTest
        VARIABLE_PREFIX "Kokkos_ENABLE"
        SPACES_CHECK    "THREADS;SERIAL;OPENMP;CUDA;HIP"
        SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
        NUM_MPI_PROCS   1
    )

    IF (Stokhos_ENABLE_KokkosKernels)
        STOKHOS_ADD_EXECUTABLE_AND_TEST(
            TEST_NAME       KokkosCrsMatrixUQPCEUnitTest
            VARIABLE_PREFIX "HAVE_TPETRA"
            SPACES_CHECK    "PTHREAD;SERIAL;OPENMP;CUDA;HIP"
            SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
            NUM_MPI_PROCS   1
        )

      IF (Stokhos_ENABLE_Tpetra)
          STOKHOS_ADD_EXECUTABLE_AND_TEST(
              TEST_NAME       TpetraCrsMatrixUQPCEUnitTest
              VARIABLE_PREFIX "HAVE_TPETRA"
              SPACES_CHECK    "PTHREAD;SERIAL;OPENMP;CUDA;HIP"
              SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
              NUM_MPI_PROCS   4
          )
      ENDIF()

    ENDIF()

  ENDIF()

  IF(Stokhos_ENABLE_Ensemble_Scalar_Type)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoMPVectorUnitTest
      SOURCES Stokhos_SacadoMPVectorUnitTest.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoMPVectorUnitTestMaskTraits
      SOURCES Stokhos_SacadoMPVectorUnitTest_MaskTraits.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoMPVectorSerializationTests
      SOURCES Stokhos_SacadoMPVectorSerializationTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      SacadoMPVectorCommTests
      SOURCES Stokhos_SacadoMPVectorCommTests.cpp ${UTILS_SOURCES}
      ARGS
      COMM serial mpi
      NUM_MPI_PROCS 1
      STANDARD_PASS_OUTPUT
      )

    STOKHOS_ADD_EXECUTABLE_AND_TEST(
        TEST_NAME       KokkosViewMPVectorUnitTest
        VARIABLE_PREFIX "Kokkos_ENABLE"
        SPACES_CHECK    "THREADS;SERIAL;OPENMP;CUDA;HIP"
        SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
        NUM_MPI_PROCS   1
    )

    IF (Stokhos_ENABLE_KokkosKernels)

        STOKHOS_ADD_EXECUTABLE_AND_TEST(
            TEST_NAME       KokkosCrsMatrixMPVectorUnitTest
            VARIABLE_PREFIX "HAVE_TPETRA"
            SPACES_CHECK    "PTHREAD;SERIAL;OPENMP;CUDA;HIP"
            SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
            NUM_MPI_PROCS   1
        )

        STOKHOS_ADD_EXECUTABLE_AND_TEST(
            TEST_NAME       KokkosViewFadMPVectorUnitTest
            VARIABLE_PREFIX "HAVE_TPETRA"
            SPACES_CHECK    "PTHREAD;SERIAL;OPENMP;CUDA;HIP"
            SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
            NUM_MPI_PROCS   1
        )

        IF (Stokhos_ENABLE_Tpetra)
            STOKHOS_ADD_EXECUTABLE_AND_TEST(
                TEST_NAME       TpetraCrsMatrixMPVectorUnitTest
                VARIABLE_PREFIX "HAVE_TPETRA"
                SPACES_CHECK    "PTHREAD;SERIAL;OPENMP;CUDA;HIP"
                SPACES_KOKKOS   "Threads;Serial;OpenMP;Cuda;HIP"
                NUM_MPI_PROCS   4
            )
        ENDIF()

    ENDIF()
  ENDIF()

ENDIF()

# Need to fix these tests so they don't require Epetra
IF (Stokhos_ENABLE_EpetraExt AND Stokhos_ENABLE_KokkosContainers)

  IF (Kokkos_ENABLE_SERIAL)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      KokkosArrayKernelsUnitTest_Serial
      SOURCES Stokhos_KokkosArrayKernelsUnitTest.hpp
             Stokhos_KokkosArrayKernelsUnitTestDecl.hpp
             Stokhos_KokkosArrayKernelsUnitTest_Host.hpp
             Stokhos_KokkosArrayKernelsUnitTest_Serial.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      )
  ENDIF()

  IF (Kokkos_ENABLE_THREADS)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      KokkosArrayKernelsUnitTest_Threads
      SOURCES Stokhos_KokkosArrayKernelsUnitTest.hpp
              Stokhos_KokkosArrayKernelsUnitTestDecl.hpp
              Stokhos_KokkosArrayKernelsUnitTest_Host.hpp
              Stokhos_KokkosArrayKernelsUnitTest_Threads.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      RUN_SERIAL
      )
  ENDIF()

  IF (Kokkos_ENABLE_OPENMP)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      KokkosArrayKernelsUnitTest_OpenMP
      SOURCES Stokhos_KokkosArrayKernelsUnitTest.hpp
              Stokhos_KokkosArrayKernelsUnitTestDecl.hpp
              Stokhos_KokkosArrayKernelsUnitTest_Host.hpp
              Stokhos_KokkosArrayKernelsUnitTest_OpenMP.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      RUN_SERIAL
      )
  ENDIF()

  IF (Kokkos_ENABLE_CUDA AND Stokhos_ENABLE_CUSPARSE)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      KokkosArrayKernelsUnitTest_Cuda
      SOURCES Stokhos_KokkosArrayKernelsUnitTest.hpp
              Stokhos_KokkosArrayKernelsUnitTestDecl.hpp
              Stokhos_KokkosArrayKernelsUnitTest_Cuda.cpp
      COMM serial mpi
      STANDARD_PASS_OUTPUT
      NUM_MPI_PROCS 1
      RUN_SERIAL
      )
  ENDIF()

ENDIF()

IF (Stokhos_ENABLE_Tpetra)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    TpetraMatVec
    SOURCES tpetra_mat_vec.cpp
    COMM serial mpi
    STANDARD_PASS_OUTPUT
    NUM_MPI_PROCS 4
    RUN_SERIAL
    )
ENDIF()
