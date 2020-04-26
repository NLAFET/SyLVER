*****
C API
*****

.. code-block:: C
                
   #include "sylver/sylver.h"

=======
General
=======

.. c:function:: void sylver_init(int ncpu, int ngpu)

   Initialization routine which should be called before any other
   routine within SyLVER. The number of CPUs and GPUs involved in the
   computations should be passed to this routine.

   :param ncpu: number of CPUs to be used in the execution of SyLVER
      routines.
   :param ngpu: number of GPUs to be used in the execution of SyLVER
      routines. Note that if CUDA is not enabled during the
      compilation, this value will be ignored.
   
.. c:function:: void sylver_finalize()

   SyLVER termination routine which should be called once all the
   desired operations have been performed.

.. c:function:: void sylver_default_options(sylver_options_t *options)

   Intialises members of options structure to default values.

   :param options: Structure to be initialised.

======
SpLDLT
======

.. note::
   
   For the most efficient use of the package, CSC format should be
   used without checking.

.. c:function:: void spldlt_analyse(int n, int *order, long const* ptr, int const* row, double const* val, void **akeep, bool check, sylver_options_t const* options, sylver_inform_t *inform)

   Perform the analyse (symbolic) phase of the factorization for a
   matrix supplied in `Compressed Sparse Column (CSC) format
   <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_. The
   resulting symbolic factors stored in :c:type:`akeep` should be
   passed unaltered in the subsequent calls to
   :c:func:`spldlt_factorize()`.

   :param n: number of columns in :math:`A`.
   :param order[]: may be `NULL`; otherwise must be an array of size `n`
      used on entry a user-supplied ordering
      (:c:type:`options.ordering=0 <sylver_options_t.ordering>`). On
      return, the actual ordering used.
   :param ptr[n+1]: column pointers for :math:`A` (see `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :param row[ptr[n]]: row indices for :math:`A` (see
      `CSC format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_).
   :param val: may be `NULL`; otherwise must be an array of size
      `ptr[n]` containing non-zero values for :math:`A` (see `CSC
      format
      <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_). Only
      used if a matching-based ordering is requested.
   :param akeep: returns symbolic factorization, to be passed
      unchanged to subsequent routines.
   :param check: if true, matrix data is checked. Out-of-range entries
      are dropped and duplicate entries are summed.
   :param options: specifies algorithm options to be used (see
      :c:type:`sylver_options_t`).
   :param inform: returns information about the execution of the
      routine (see :c:type:`sylver_inform_t`).

   .. note::

      If a user-supplied ordering is used, it may be altered by this
      routine, with the altered version returned in `order[]`. This
      version will be equivalent to the original ordering, except that
      some supernodes may have been amalgamated, a topologic ordering
      may have been applied to the assembly tree and the order of
      columns within a supernode may have been adjusted to improve
      cache locality.
      
.. c:function:: void spldlt_factorize(bool posdef, long const* ptr, int const* row, double const* val, double *scale, void *akeep, void **fkeep, sylver_options_t const* options, sylver_inform_t *inform)

   :param posdef: true if matrix is positive-definite
   :param ptr: may be `NULL`; otherwise a length `n+1` array of column
      pointers for :math:`A`, only required if :f:type:`akeep` was
      obtained by running :c:func:`spldlt_analyse()` with
      `check=true`, in which case it must be unchanged since that
      call.
   :param row: may be `NULL`; otherwise a length `ptr[n]` array of row
      indices for :math:`A`, only required if :f:type:`akeep` was
      obtained by running :c:func:`spldlt_analyse()` with
      `check=true`, in which case it must be unchanged since that
      call.
   :param val[]: non-zero values for :math:`A` in same format as for
      the call to :c:func:`spldlt_analyse()`.
   :param scale: may be `NULL`; otherwise a length `n` array for
      diagonal scaling. `scale[i-1]` contains entry :math:`S_ii` of
      :math:`S`. Must be supplied by user on entry if
      :c:member:`options.scaling=0 <sylver_options_t.scaling>`
      (user-supplied scaling). On exit, returns scaling used.
   :param akeep: symbolic factorization returned by preceding call to
      :c:func:`spldlt_analyse()`.
   :param fkeep: returns numeric factorization, to be passed unchanged
      to subsequent routines.
   :param options: specifies algorithm options to be used
      (see :c:type:`sylver_options_t`).
   :param inform: returns information about the execution of the routine
      (see :c:type:`sylver_inform_t`).

                
.. c:function:: void spldlt_solve(int job, int nrhs, double *x, int ldx, void *akeep, void *fkeep, sylver_options_t const* options, sylver_inform_t *inform)

   Solve (for :math:`nrhs` right-hand sides) one of the following
   equations:

   +---------------+--------------------------+
   | `job`         | Equation solved          |
   +===============+==========================+
   | 0             | :math:`AX=B`             |
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

   :param job: specifies equation to solve, as per above table.
   :param nrhs: number of right-hand sides.
   :param x[ldx*nrhs]: right-hand sides :math:`B` on entry, solutions
      :math:`X` on exit. The `i`-th entry of right-hand side `j` is in
      position `x[j*ldx+i]`.
   :param ldx: leading dimension of `x`.
   :param akeep: symbolic factorization returned by preceding call to
      :c:func:`spldlt_analyse()`.
   :param fkeep: numeric factorization returned by preceding call to
      :c:func:`spldlt_factor()`.
   :param options: specifies algorithm options to be used (see
      :c:type:`sylver_options_t`).
   :param inform: returns information about the execution of the
      routine (see :c:type:`sylver_inform_t`).

.. c:function:: void spldlt_free_akeep(void **akeep)

   Frees memory and resources associated with :c:type:`akeep`.

   :param akeep: symbolic factors to be freed.

.. c:function:: int spldlt_free_fkeep(void **fkeep)

   Frees memory and resources associated with :c:type:`fkeep`.

   :param fkeep: numeric factors to be freed.
