***
API
***

In the below, all reals are double precision unless otherwise indicated.

=======
General
=======

.. f:subroutine:: sylver_init(ncpu, ngpu)

   Initialization routine which should be called before any other
   routine within SyLVER. The number of CPUs and GPUs involved in the
   computations should be passed to this routine.

   :p integer ncpu [in]: number of CPUs to be used in the
                         execution of SyLVER routines.
   :p integer ngpu [in]: number of GPUs to be used in the execution of
                         SyLVER routines. Note that if CUDA is not
                         enabled during the compilation, this value
                         will be ignored.

.. f:subroutine:: sylver_finalize()

   SyLVER termination routine which should be called once all the
   desired operations have been performed.
                  
======
SpLDLT
======

.. f:subroutine:: spldlt_analyse(akeep,n,ptr,row,options,inform [,order,val, ncpu, ngpu, check])

   Perform the analyse (symbolic) phase of the factorization for a
   matrix supplied in `Compressed Sparse Column (CSC) format
   <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_. The
   resulting symbolic factors stored in :f:type:`spldlt_akeep` should be passed
   unaltered in the subsequent calls to ssids_factor().

   :p spldlt_akeep akeep [out]: returns symbolic factorization, to be
      passed unchanged to subsequent routines.
   :p integer n [in]: number of columns in :math:`A`.
   :p long ptr(n+1) [in]: column pointers for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :p integer row(ptr(n+1)-1) [in]: row indices for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :p sylver_options options [in]: specifies algorithm options to be used
      (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).                                    
   :o integer order(n) [inout]: on entry a user-supplied ordering
      (options%ordering=0). On return, the actual ordering used (if present).
   :o real val(ptr(n+1)-1) [in]: non-zero values for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_). Only
      used if a matching-based ordering is requested.
   :o integer ncpu [in]: Number of CPU available for the execution. If
                         absent, the value of ncpu passed to the
                         :f:subr:`sylver_init()` routine is used
                         instead.
   :o integer ncpu [in]: Number of GPU available for the
                         execution. This value is ignored if CUDA is
                         not enabled during the compilation and if
                         absent, the value of ncpu passed to the
                         :f:subr:`sylver_init()` routine is used
                         instead.
   :o logical check [in]: if true, matrix data is
      checked. Out-of-range entries are dropped and duplicate entries
      are summed.
   
   .. note::

      If a user-supplied ordering is used, it may be altered by this
      routine, with the altered version returned in order(:). This
      version will be equivalent to the original ordering, except that
      some supernodes may have been amalgamated, a topological
      ordering may have been applied to the assembly tree and the
      order of columns within a supernode may have been adjusted to
      improve cache locality.

.. f:subroutine::  spldlt_factorize(akeep,fkeep,posdef,val,options,inform[,scale,ptr,row])

   :p spldlt_akeep akeep [in]: symbolic factorization returned by
      preceding call to :f:subr:`spldlt_analyse()`.
   :p spldlt_akeep akeep [out]: returns numeric factorization, to be
      passed unchanged to subsequent routines.
   :p logical posdef [in]: true if matrix is positive-definite.
   :p real val(*) [in]: non-zero values for :math:`A` in same format
      as for the call to :f:subr:`spldlt_analyse()`.
   :p sylver_options options [in]: specifies algorithm options to be
      used (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).
   :o real scale(n) [inout]: diagonal scaling. scale(i) contains entry
      :math:`S_{ii}` of :math:`S`. Must be supplied by user if
      ``options%scaling=0`` (user-supplied scaling). On exit, return scaling
      used.
   :o integer(long) ptr(n+1) [in]: column pointers for :math:`A`, only
      required if scaling is required (options%scaling > 0) expect in
      the case where matching-based ordering is done (options%scaling = 3)
   :o integer row(ptr(n+1)-1) [in]: row indices for :math:`A`, only
      required if scaling is required (options%scaling > 0) expect in
      the case where matching-based ordering is done (options%scaling
      = 3)

.. f:subroutine:: spldlt_solve(akeep,fkeep,nrhs,x,ldx,options,inform[,job])

   Solve (for multiple right-hand sides) one of the following equations:

   +---------------+--------------------------+
   | `job`         | Equation solved          |
   +===============+==========================+
   | 0 (or absent) | :math:`AX=B`             |
   +---------------+--------------------------+
   | 1             | :math:`PLX=SB`           |
   +---------------+--------------------------+
   | 2             | :math:`DX=B`             |
   +---------------+--------------------------+
   | 3             | :math:`(PL)^TS^{-1}X=B`  |
   +---------------+--------------------------+
   | 4             | :math:`D(PL)^TS^{-1}X=B` |
   +---------------+--------------------------+

   Recall :math:`A` has been factorized as either:
   
   * :math:`SAS = (PL)(PL)^T~` (positive-definite case); or
   * :math:`SAS = (PL)D(PL)^T` (indefinite case).

   :p spldlt_akeep akeep [in]: symbolic factorization returned by preceding
      call to :f:subr:`spldlt_analyse()`
   :p spldlt_fkeep fkeep [in]: numeric factorization returned by preceding
      call to :f:subr:`spldlt_factor()`.
   :p integer nrhs [in]: number of right-hand sides.
   :p real x(ldx,nrhs) [inout]: right-hand sides :math:`B` on entry,
      solutions :math:`X` on exit.
   :p integer ldx [in]: leading dimension of :f:type:`x`.
   :p sylver_options options [in]: specifies algorithm options to be used
      (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).                                    
   :o integer job [in]: specifies equation to solve, as per above table.

====
SpLU
====


.. f:subroutine:: splu_analyse(akeep,n,ptr,row,options,inform, ncpu[,order,val])

   Perform the analyse (symbolic) phase of the factorization for a
   matrix supplied in `Compressed Sparse Column (CSC) format
   <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_. The
   resulting symbolic factors stored in `splu_akeep` should be passed
   unaltered in the subsequent calls to ssids_factor().

   :p splu_akeep akeep [out]: returns symbolic factorization, to be
      passed unchanged to subsequent routines.
   :p integer n [in]: number of columns in :math:`A`.
   :p long ptr(n+1) [in]: column pointers for :math:`A` (see `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :p integer row(ptr(n+1)-1) [in]: row indices for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :p sylver_options options [in]: specifies algorithm options to be used
      (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).                                    
   :p integer ncpu [in]: Number of CPU available for the execution.
   :o integer order(n) [inout]: on entry a user-supplied ordering
      (options%ordering=0). On return, the actual ordering used (if present).
   :o real val(ptr(n+1)-1) [in]: non-zero values for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_). Only
      used if a matching-based ordering is requested.

.. f:subroutine::  splu_factorize(akeep,fkeep,posdef,val,options,inform[,scale,ptr,row])

   :p splu_akeep akeep [in]: symbolic factorization returned by
      preceding call to :f:subr:`splu_analyse()`.
   :p splu_akeep akeep [out]: returns numeric factorization, to be
      passed unchanged to subsequent routines.
   :p logical posdef [in]: true if matrix is positive-definite.
   :p real val(*) [in]: non-zero values for :math:`A` in same format
      as for the call to :f:subr:`splu_analyse()`.
   :p sylver_options options [in]: specifies algorithm options to be
      used (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).
   :o real scale(n) [inout]: diagonal scaling. scale(i) contains entry
      :math:`S_{ii}` of :math:`S`. Must be supplied by user if
      ``options%scaling=0`` (user-supplied scaling). On exit, return scaling
      used.

.. f:subroutine:: splu_solve(akeep,fkeep,nrhs,x,ldx,options,inform[,job])

   Solve (for multiple right-hand sides) one of the following equations:

   +---------------+--------------------------+
   | `job`         | Equation solved          |
   +===============+==========================+
   | 0 (or absent) | :math:`AX=B`             |
   +---------------+--------------------------+
   | 1             | :math:`PLX=SB`           |
   +---------------+--------------------------+
   | 2             | :math:`DX=B`             |
   +---------------+--------------------------+
   | 3             | :math:`(PL)^TS^{-1}X=B`  |
   +---------------+--------------------------+
   | 4             | :math:`D(PL)^TS^{-1}X=B` |
   +---------------+--------------------------+

   Recall :math:`A` has been factorized as:
   
   * :math:`SAS = PLUQ`
     
   Where :math:`P` and :math:`Q` are permutation matices.

   :p spldlt_akeep akeep [in]: symbolic factorization returned by preceding
      call to :f:subr:`spldlt_analyse()`
   :p spldlt_fkeep fkeep [in]: numeric factorization returned by preceding
      call to :f:subr:`spldlt_factor()`.
   :p integer nrhs [in]: number of right-hand sides.
   :p real x(ldx,nrhs) [inout]: right-hand sides :math:`B` on entry,
      solutions :math:`X` on exit.
   :p integer ldx [in]: leading dimension of :f:type:`x`.
   :p sylver_options options [in]: specifies algorithm options to be used
      (see :f:type:`sylver_options`).
   :p sylver_inform inform [out]: returns information about the
      execution of the routine (see :f:type:`sylver_inform`).                                    
   :o integer job [in]: specifies equation to solve, as per above table.
