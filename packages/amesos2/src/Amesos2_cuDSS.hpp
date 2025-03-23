// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef AMESOS2_CUDSS_HPP
#define AMESOS2_CUDSS_HPP

#include "Amesos2_SolverCore.hpp"
#include "Amesos2_SolverTraits.hpp"

namespace Amesos2 {

/**
 * \brief Amesos2 interface to cuDSS.
 *
 * \ingroup amesos2_solver_interfaces
 */
template <class Matrix, class Vector>
class cuDSS : public SolverCore<cuDSS, Matrix, Vector>
{
  friend class SolverCore<cuDSS, Matrix, Vector>;

public:
  cuDSS(
    Teuchos::RCP<const Matrix> A,
    Teuchos::RCP<Vector>       X,
    Teuchos::RCP<const Vector> B)
  : SolverCore<Amesos2::cuDSS, Matrix, Vector>(std::move(A), std::move(X), std::move(B)) {};

private:
  int preOrdering_impl() {};
  int symbolicFactorization_impl() {};
  int numericFactorization_impl() {};
  int solve_impl(
    const Teuchos::Ptr<MultiVecAdapter<Vector>> X,
    const Teuchos::Ptr<const MultiVecAdapter<Vector>> B
  ) const {};
  bool matrixShapeOK_impl() const {};
  void setParameters_impl(const Teuchos::RCP<Teuchos::ParameterList>& parameterList) {};
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters_impl() const {};
  bool loadA_impl(EPhase current_phase) {};
  bool do_optimization() const {};
};

template <>
struct solver_traits<cuDSS> {
#ifdef HAVE_TEUCHOS_COMPLEX
  using supported_scalars = Meta::make_list6<
    float, double,
    std::complex<float>, std::complex<double>,
    Kokkos::complex<float>, Kokkos::complex<double>
  >;
#else
  using supported_scalars = Meta::make_list2<float, double>;
#endif
};

template <typename Scalar, typename LocalOrdinal, typename ExecutionSpace>
struct solver_supports_matrix<cuDSS, KokkosSparse::CrsMatrix<Scalar, LocalOrdinal, ExecutionSpace>> {
  static const bool value = true;
};

} // end namespace Amesos2

#endif  // AMESOS2_CUDSS_HPP
