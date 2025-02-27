// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef TPETRA_MATRIXMATRIX_HIP_DEF_HPP
#define TPETRA_MATRIXMATRIX_HIP_DEF_HPP

#ifdef HAVE_TPETRA_INST_HIP
namespace Tpetra {
namespace MMdetails {

/*********************************************************************************************************/
// MMM KernelWrappers for Partial Specialization to HIP
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
struct KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType> {
    static inline void mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                  const std::string& label = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);



   static inline void mult_A_B_reuse_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                  const std::string& label = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

};

// Jacobi KernelWrappers for Partial Specialization to HIP
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal, class LocalOrdinalViewType>
struct KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType> {
    static inline void jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                           const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
                                                           CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                           CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                           const LocalOrdinalViewType & Acol2Brow,
                                                           const LocalOrdinalViewType & Acol2Irow,
                                                           const LocalOrdinalViewType & Bcol2Ccol,
                                                           const LocalOrdinalViewType & Icol2Ccol,
                                                           CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                           Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                           const std::string& label = std::string(),
                                                           const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

    static inline void jacobi_A_B_reuse_kernel_wrapper(Scalar omega,
                                                       const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                       const LocalOrdinalViewType & Acol2Brow,
                                                       const LocalOrdinalViewType & Acol2Irow,
                                                       const LocalOrdinalViewType & Bcol2Ccol,
                                                       const LocalOrdinalViewType & Icol2Ccol,
                                                       CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                       Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                       const std::string& label = std::string(),
                                                       const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

    static inline void jacobi_A_B_newmatrix_KokkosKernels(Scalar omega,
                                                          const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
                                                          CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                          CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                          const LocalOrdinalViewType & Acol2Brow,
                                                          const LocalOrdinalViewType & Acol2Irow,
                                                          const LocalOrdinalViewType & Bcol2Ccol,
                                                          const LocalOrdinalViewType & Icol2Ccol,
                                                          CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                          Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                          const std::string& label = std::string(),
                                                          const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);
};


/*********************************************************************************************************/
// AB NewMatrix Kernel wrappers (KokkosKernels/HIP Version)
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType>::mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {


#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<TimeMonitor> MM = rcp(new TimeMonitor(*(TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix HIPWrapper")))));
#endif
  // Node-specific code
  typedef Kokkos::Compat::KokkosHIPWrapperNode Node;
  std::string nodename("HIP");

  // Lots and lots of typedefs
  using Teuchos::RCP;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_device_type KCRS;
  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::row_map_type::const_type  c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  //typedef typename graph_t::row_map_type::const_type lno_view_t_const;

  // Options
  int team_work_size = 16;  // Defaults to 16 as per Deveci 12/7/16 - csiefer
  std::string myalg("SPGEMM_KK_MEMORY");
  if(!params.is_null()) {
    if(params->isParameter("hip: algorithm"))
      myalg = params->get("hip: algorithm",myalg);
    if(params->isParameter("hip: team work size"))
      team_work_size = params->get("hip: team work size",team_work_size);
  }

  // KokkosKernelsHandle
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
       typename lno_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
       typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space > KernelHandle;

  // Grab the  Kokkos::SparseCrsMatrices
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixDevice();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixDevice();

  c_lno_view_t Arowptr = Amat.graph.row_map,
               Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries,
                       Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values,
                      Bvals = Bmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixDevice();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }


  // Get the algorithm mode
  std::string alg = nodename+std::string(" algorithm");
  //  printf("DEBUG: Using kernel: %s\n",myalg.c_str());
  if(!params.is_null() && params->isParameter(alg)) myalg = params->get(alg,myalg);
  KokkosSparse::SPGEMMAlgorithm alg_enum = KokkosSparse::StringToSPGEMMAlgorithm(myalg);

  // Merge the B and Bimport matrices
  const KCRS Bmerged = Tpetra::MMdetails::merge_matrices(Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C.getColMap()->getLocalNumElements());

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix HIPCore"))));
#endif

  // Do the multiply on whatever we've got
  typename KernelHandle::nnz_lno_t AnumRows = Amat.numRows();
  typename KernelHandle::nnz_lno_t BnumRows = Bmerged.numRows();
  typename KernelHandle::nnz_lno_t BnumCols = Bmerged.numCols();

  lno_view_t      row_mapC (Kokkos::ViewAllocateWithoutInitializing("non_const_lnow_row"), AnumRows + 1);
  lno_nnz_view_t  entriesC;
  scalar_view_t   valuesC;
  KernelHandle kh;
  kh.create_spgemm_handle(alg_enum);
  kh.set_team_work_size(team_work_size);

  KokkosSparse::Experimental::spgemm_symbolic(&kh,AnumRows,BnumRows,BnumCols,Amat.graph.row_map,Amat.graph.entries,false,Bmerged.graph.row_map,Bmerged.graph.entries,false,row_mapC);

  size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
  if (c_nnz_size){
    entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
    valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
  }
  KokkosSparse::Experimental::spgemm_numeric(&kh,AnumRows,BnumRows,BnumCols,Amat.graph.row_map,Amat.graph.entries,Amat.values,false,Bmerged.graph.row_map,Bmerged.graph.entries,Bmerged.values,false,row_mapC,entriesC,valuesC);
  kh.destroy_spgemm_handle();

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix HIPSort"))));
#endif

  // Sort & set values
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
  C.setAllValues(row_mapC,entriesC,valuesC);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix HIPESFC"))));
#endif

  // Final Fillcomplete
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  RCP<const Export<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > dummyExport;
  C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
}


/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType>::mult_A_B_reuse_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & targetMapToOrigRow_dev,
                                                                                               const LocalOrdinalViewType & targetMapToImportRow_dev,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol_dev,
                                                                                               const LocalOrdinalViewType & Icol2Ccol_dev,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

  // FIXME: Right now, this is a cut-and-paste of the serial kernel
  typedef Kokkos::Compat::KokkosHIPWrapperNode Node;

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Reuse SerialCore"))));
  Teuchos::RCP<Teuchos::TimeMonitor> MM2;
#endif
  using Teuchos::RCP;
  using Teuchos::rcp;


  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>     map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           targetMapToOrigRow_dev);
  auto targetMapToImportRow =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           targetMapToImportRow_dev);
  auto Bcol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           Bcol2Ccol_dev);
  auto Icol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           Icol2Ccol_dev);

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getLocalNumRows();
  size_t n = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS & Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map,
               Browptr = Bmat.graph.row_map,
               Crowptr = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries,
                       Bcolind = Bmat.graph.entries,
                       Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2 = Teuchos::null; MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix SerialCore - Compare"))));
#endif

  // Classic csr assembly (low memory edition)
  // mfh 27 Sep 2016: The c_status array is an implementation detail
  // of the local sparse matrix-matrix multiply routine.

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i+1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    for (size_t k = Arowptr[i]; k < Arowptr[i+1]; k++) {
      LO Aik  = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk+1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
            "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = Teuchos::as<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik+1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
            "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Ivals[j];
        }
      }
    }
  }

  C.fillComplete(C.getDomainMap(), C.getRangeMap());
}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType>::jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                                                               const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

#ifdef HAVE_TPETRA_MMM_TIMINGS
    std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
    using Teuchos::TimeMonitor;
    Teuchos::RCP<TimeMonitor> MM;
#endif

  // Node-specific code
  using Teuchos::RCP;

  // Options
  //int team_work_size = 16;  // Defaults to 16 as per Deveci 12/7/16 - csiefer // unreferenced
  std::string myalg("KK");
  if(!params.is_null()) {
    if(params->isParameter("hip: jacobi algorithm"))
      myalg = params->get("hip: jacobi algorithm",myalg);
  }

  if(myalg == "MSAK") {
    ::Tpetra::MatrixMatrix::ExtraKernels::jacobi_A_B_newmatrix_MultiplyScaleAddKernel(omega,Dinv,Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C,Cimport,label,params);
  }
  else if(myalg == "KK") {
    jacobi_A_B_newmatrix_KokkosKernels(omega,Dinv,Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C,Cimport,label,params);
  }
  else {
    throw std::runtime_error("Tpetra::MatrixMatrix::Jacobi newmatrix unknown kernel");
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix HIPESFC"))));
#endif

  // Final Fillcomplete
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));

  // NOTE: MSAK already fillCompletes, so we have to check here
  if(!C.isFillComplete()) {
    RCP<const Export<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > dummyExport;
    C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
  }

}



/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType>::jacobi_A_B_reuse_kernel_wrapper(Scalar omega,
                                                                                               const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & targetMapToOrigRow_dev,
                                                                                               const LocalOrdinalViewType & targetMapToImportRow_dev,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol_dev,
                                                                                               const LocalOrdinalViewType & Icol2Ccol_dev,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

  // FIXME: Right now, this is a cut-and-paste of the serial kernel
  typedef Kokkos::Compat::KokkosHIPWrapperNode Node;

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse HIPCore"))));
  Teuchos::RCP<Teuchos::TimeMonitor> MM2;
#endif
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>     map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           targetMapToOrigRow_dev);
  auto targetMapToImportRow =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           targetMapToImportRow_dev);
  auto Bcol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           Bcol2Ccol_dev);
  auto Icol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                           Icol2Ccol_dev);


  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getLocalNumRows();
  size_t n = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS & Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map, Crowptr = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries, Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }

  // Jacobi-specific inner stuff
  auto Dvals =
       Dinv.template getLocalView<scalar_memory_space>(Access::ReadOnly);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2 = Teuchos::null; MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse HIPCore - Compare"))));
#endif

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {

    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i+1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    SC minusOmegaDval = -omega*Dvals(i,0);

    // Entries of B
    for (size_t j = Browptr[i]; j < Browptr[i+1]; j++) {
      Scalar Bval = Bvals[j];
      if (Bval == SC_ZERO)
        continue;
      LO Bij = Bcolind[j];
      LO Cij = Bcol2Ccol[Bij];

      TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
        std::runtime_error, "Trying to insert a new entry into a static graph");

      Cvals[c_status[Cij]] = Bvals[j];
    }

    // Entries of -omega * Dinv * A * B
    for (size_t k = Arowptr[i]; k < Arowptr[i+1]; k++) {
      LO Aik  = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk+1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = Teuchos::as<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik+1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Ivals[j];
        }
      }
    }
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2= Teuchos::null;
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse ESFC"))));
#endif

  C.fillComplete(C.getDomainMap(), C.getRangeMap());

}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode,LocalOrdinalViewType>::jacobi_A_B_newmatrix_KokkosKernels(Scalar omega,
												const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> & Dinv,
											        CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Aview,
												CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& Bview,
												const LocalOrdinalViewType & Acol2Brow,
												const LocalOrdinalViewType & Acol2Irow,
												const LocalOrdinalViewType & Bcol2Ccol,
												const LocalOrdinalViewType & Icol2Ccol,
												CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosHIPWrapperNode>& C,
												Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > Cimport,
												const std::string& label,
												const Teuchos::RCP<Teuchos::ParameterList>& params) {

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<TimeMonitor> MM;
#endif

  // Check if the diagonal entries exist in debug mode
  const bool debug = Tpetra::Details::Behavior::debug();
  if(debug) {

    auto rowMap = Aview.origMatrix->getRowMap();
    Tpetra::Vector<Scalar> diags(rowMap);
    Aview.origMatrix->getLocalDiagCopy(diags);
    size_t diagLength = rowMap->getLocalNumElements();
    Teuchos::Array<Scalar> diagonal(diagLength);
    diags.get1dCopy(diagonal());

    for(size_t i = 0; i < diagLength; ++i) {
      TEUCHOS_TEST_FOR_EXCEPTION(diagonal[i] == Teuchos::ScalarTraits<Scalar>::zero(),
				 std::runtime_error,
				 "Matrix A has a zero/missing diagonal: " << diagonal[i] << std::endl <<
				 "KokkosKernels Jacobi-fused SpGEMM requires nonzero diagonal entries in A" << std::endl);
    }
  }

  // Usings
  using device_t = typename Kokkos::Compat::KokkosHIPWrapperNode::device_type;
  using matrix_t = typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode>::local_matrix_device_type;
  using graph_t = typename matrix_t::StaticCrsGraphType;
  using lno_view_t = typename graph_t::row_map_type::non_const_type;
  using c_lno_view_t = typename graph_t::row_map_type::const_type;
  using lno_nnz_view_t = typename graph_t::entries_type::non_const_type;
  using scalar_view_t = typename matrix_t::values_type::non_const_type;

  // KokkosKernels handle
  using handle_t = typename KokkosKernels::Experimental::KokkosKernelsHandle<
    typename lno_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
    typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space >;

  // Get the rowPtr, colInd and vals of importMatrix
  c_lno_view_t Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixDevice();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }

  // Merge the B and Bimport matrices
  const matrix_t Bmerged = Tpetra::MMdetails::merge_matrices(Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C.getColMap()->getLocalNumElements());

  // Get the properties and arrays of input matrices
  const matrix_t & Amat = Aview.origMatrix->getLocalMatrixDevice();
  const matrix_t & Bmat = Bview.origMatrix->getLocalMatrixDevice();

  typename handle_t::nnz_lno_t AnumRows = Amat.numRows();
  typename handle_t::nnz_lno_t BnumRows = Bmerged.numRows();
  typename handle_t::nnz_lno_t BnumCols = Bmerged.numCols();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmerged.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmerged.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmerged.values;

  // Arrays of the output matrix
  lno_view_t row_mapC (Kokkos::ViewAllocateWithoutInitializing("non_const_lnow_row"), AnumRows + 1);
  lno_nnz_view_t entriesC;
  scalar_view_t valuesC;

  // Options
  int team_work_size = 16;
  std::string myalg("SPGEMM_KK_MEMORY");
  if(!params.is_null()) {
    if(params->isParameter("hip: algorithm"))
      myalg = params->get("hip: algorithm",myalg);
    if(params->isParameter("hip: team work size"))
      team_work_size = params->get("hip: team work size",team_work_size);
  }

  // Get the algorithm mode
  std::string nodename("HIP");
  std::string alg = nodename + std::string(" algorithm");
  if(!params.is_null() && params->isParameter(alg)) myalg = params->get(alg,myalg);
  KokkosSparse::SPGEMMAlgorithm alg_enum = KokkosSparse::StringToSPGEMMAlgorithm(myalg);


  // KokkosKernels call
  handle_t kh;
  kh.create_spgemm_handle(alg_enum);
  kh.set_team_work_size(team_work_size);

  KokkosSparse::Experimental::spgemm_symbolic(&kh, AnumRows, BnumRows, BnumCols,
					      Arowptr, Acolind, false,
					      Browptr, Bcolind, false,
					      row_mapC);

  size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
  if (c_nnz_size){
    entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
    valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
  }

  KokkosSparse::Experimental::spgemm_jacobi(&kh, AnumRows, BnumRows, BnumCols,
					    Arowptr, Acolind, Avals, false,
					    Browptr, Bcolind, Bvals, false,
					    row_mapC, entriesC, valuesC,
					    omega, Dinv.getLocalViewDevice(Access::ReadOnly));
  kh.destroy_spgemm_handle();

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix HIPSort"))));
#endif

  // Sort & set values
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
  C.setAllValues(row_mapC,entriesC,valuesC);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix HIPESFC"))));
#endif

  // Final Fillcomplete
  Teuchos::RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosHIPWrapperNode> > dummyExport;
  C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
}

  }//MMdetails
}//Tpetra

#endif//HIP

#endif
