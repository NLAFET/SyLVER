*******
API (C)
*******

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

   Solve (for `nrhs` right-hand sides) one of the following equations:

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

==========
Data types
==========

-------
Options
-------

.. c:type:: struct sylver_options_t

   The data type :c:type:`sylver_options_t` is used to specify the
   options used within ``SyLVER``. The components, that are
   automatically given default values in the definition of the type,
   are:

   .. c:member:: int print_level
   
      Level of printing:

      +---------------+-------------------------------------------------+
      | < 0           | No printing.                                    |
      +---------------+-------------------------------------------------+
      | = 0 (default) | Error and warning messages only.                |
      +---------------+-------------------------------------------------+
      | = 1           | As 0, plus basic diagnostic printing.           |
      +---------------+-------------------------------------------------+
      | > 1           | As 1, plus some additional diagnostic printing. |
      +---------------+-------------------------------------------------+

      The default is 0.

   .. c:member:: int unit_diagnostics
   
      Fortran unit number for diagnostics printing.
      Printing is suppressed if <0.

      The default is 6 (stdout).

   .. c:member:: int unit_error
   
      Fortran unit number for printing of error messages.
      Printing is suppressed if <0.
      
      The default is 6 (stdout).

   .. c:member:: int unit_warning
   
      Fortran unit number for printing of warning messages.
      Printing is suppressed if <0.

      The default is 6 (stdout).
      
   .. c:member:: int ordering
   
      Ordering method to use in analyse phase:

      +-------------+---------------------------------------------------------+
      | 0           | User-supplied ordering is used (`order` argument to     |
      |             | :c:func:`sylver_analyse()`).                            |
      +-------------+---------------------------------------------------------+
      | 1 (default) | METIS ordering with default settings.                   |
      +-------------+---------------------------------------------------------+
      | 2           | Matching-based elimination ordering is computed (the    |
      |             | Hungarian algorithm is used to identify large           |
      |             | off-diagonal entries. A restricted METIS ordering is    |
      |             | then used that forces these on to the subdiagonal).     |
      |             |                                                         |
      |             | **Note:** This option should only be chosen for         |
      |             | indefinite systems. A scaling is also computed that may |
      |             | be used in :c:func:`sylver_factor()` (see               |
      |             | :c:member:`scaling <sylver_options_t.scaling>`          |
      |             | below).                                                 |
      +-------------+---------------------------------------------------------+

      The default is 1.

   .. c:member:: int nemin
   
      Supernode amalgamation threshold. Two neighbours in the elimination tree
      are merged if they both involve fewer than `nemin` eliminations.
      The default is used if `nemin<1`.
      The default is 8.

   .. c:member:: bool prune_tree
   
      If true, prune the elimination tree to better exploit data
      locality in the parallel factorization.

      The default is `true`.

   .. c:member:: long min_gpu_work
   
      Minimum number of flops in subtree before scheduling on GPU.

      Default is `5e9`.

   .. c:member:: int scaling
   
      Scaling algorithm to use:

      +---------------+-------------------------------------------------------+
      | <=0 (default) | No scaling (if ``scale[]`` is not present on call to  |
      |               | :c:func:`spldlt_factor()`, or user-supplied           |
      |               | scaling (if ``scale[]`` is present).                  |
      +---------------+-------------------------------------------------------+
      | =1            | Compute using weighted bipartite matching via the     |
      |               | Hungarian Algorithm (MC64 algorithm).                 |
      +---------------+-------------------------------------------------------+
      | =2            | Compute using a weighted bipartite matching via the   |
      |               | Auction Algorithm (may be lower quality than that     |
      |               | computed using the Hungarian Algorithm, but can be    |
      |               | considerably faster).                                 |
      +---------------+-------------------------------------------------------+
      | =3            | Use matching-based ordering generated during the      |
      |               | analyse phase using :c:member:`options.ordering=2     |
      |               | <sylver_options_t.ordering>`. The scaling             |
      |               | will be the same as that generated with               |
      |               | :c:member:`options.scaling=1                          |
      |               | <sylver_options_t.scaling>`                           |
      |               | if the matrix values have not changed. This option    |
      |               | will generate an error if a matching-based ordering   |
      |               | was not used during analysis.                         |
      +---------------+-------------------------------------------------------+
      | >=4           | Compute using the norm-equilibration algorithm of     |
      |               | Ruiz (see :doc:`scaling`).                            |
      +---------------+-------------------------------------------------------+

      The default is 0.

   .. c:member:: int pivot_method
   
      Pivot method to be used on CPU, one of:

      +-------------+----------------------------------------------------------+
      | 2 (default) | Block a posteori pivoting. A failed pivot only requires  |
      |             | recalculation of entries within its own block column.    |
      +-------------+----------------------------------------------------------+
      | 3           | Threshold partial pivoting. Not parallel.                |
      +-------------+----------------------------------------------------------+

      Default is `2`.

   .. c:member:: double small

      Threshold below which an entry is treated as equivalent to
      `0.0`.

      The default is `1e-20`.

   .. c:member:: double u
   
      Relative pivot threshold used in symmetric indefinite
      case. Values outside of the range :math:`[0,0.5]` are treated as
      the closest value in that range.

      The default is `0.01`.

   .. c:member:: long small_subtree_threshold
   
      Maximum number of flops in a subtree treated as a single
      task.

      The default is `4e6`.

   .. c:member:: int nb
   
      Block size to use for parallelization of large nodes on CPU
      resources.

      Default is `256`.

   .. c:member:: int cpu_topology
   
      +-------------+----------------------------------------------------------+
      | 1 (default) | Automatically chose the CPU tology depending on the      |
      |             | underlying architecture.                                 |
      +-------------+----------------------------------------------------------+
      | 2           | Assume flat topology and in particular ignore NUMA       |
      |             | structure.                                               |
      +-------------+----------------------------------------------------------+
      | 3           | Use NUMA structure of underlying architecture to better  |
      |             | exploit data locality in the parallel execution          |
      +-------------+----------------------------------------------------------+

      Default is `1`.

   .. c:member:: bool action
   
      Continue factorization of singular matrix on discovery of zero
      pivot if `true` (a warning is issued), or abort if `false`.

      The default is `true`.

   .. c:member:: bool use_gpu
   
      Use an NVIDIA GPU if present.

      Default is `true`.

   .. c:member:: float gpu_perf_coeff
   
      GPU perfromance coefficient. How many times faster a GPU is than
      CPU at factoring a subtree.

      Default is `1.0`.

-----------
Information
-----------
      
.. c:type:: struct sylver_inform_t

   Used to return information about the progress and needs of the
   algorithm.

   .. c:member:: int flag
      
      Exit status of the algorithm (see table below).

   .. c:member:: int matrix_dup
   
      Number of duplicate entries encountered (if
      :c:func:`spldlt_analyse()` called with check=true).

   .. c:member:: int matrix_missing_diag
   
      Number of diagonal entries without an explicit value (if
      :c:func:`spldlt_analyse()` called with check=true).

   .. c:member:: int matrix_outrange
   
      Number of out-of-range entries encountered (if
      :c:func:`spldlt_analyse()` called with check=true).

   .. c:member:: int matrix_rank
   
      (Estimated) rank (structural after analyse phase, numerical
      after factorize phase).

   .. c:member:: int maxdepth
   
      Maximum depth of the assembly tree.

   .. c:member:: int maxfront
   
      Maximum front size (without pivoting after analyse phase, with
      pivoting after factorize phase).

   .. c:member:: int num_delay
   
      Number of delayed pivots. That is, the total number of
      fully-summed variables that were passed to the father node
      because of stability considerations. If a variable is passed
      further up the tree, it will be counted again.

   .. c:member:: long num_factor
   
      Number of entries in :math:`L` (without pivoting after analyse
      phase, with pivoting after factorize phase).

   .. c:member:: long num_flops
   
      Number of floating-point operations for Cholesky factorization
      (indefinte needs slightly more). Without pivoting after analyse
      phase, with pivoting after factorize phase.

   .. c:member:: int num_neg
   
      Number of negative eigenvalues of the matrix :math:`D` after
      factorize phase.

   .. c:member:: int num_sup
   
      Number of supernodes in assembly tree.

   .. c:member:: int num_two
   
      Number of :math:`2 \times 2` pivots used by the factorization
      (i.e. in the matrix :math:`D`).

   .. c:member:: int stat
      
      Fortran allocation status parameter in event of allocation error
      (0 otherwise).

   +-------------+-------------------------------------------------------------+
   | inform.flag | Return status                                               |
   +=============+=============================================================+
   | 0           | Success.                                                    |
   +-------------+-------------------------------------------------------------+
   | -1          | Error in sequence of calls (may be caused by failure of a   |
   |             | preceding call).                                            |
   +-------------+-------------------------------------------------------------+
   | -2          | n<0 or ne<1.                                                |
   +-------------+-------------------------------------------------------------+
   | -3          | Error in ptr[].                                             |
   +-------------+-------------------------------------------------------------+
   | -4          | CSC format: All variable indices in one or more columns are |
   |             | out-of-range.                                               |
   |             |                                                             |
   |             | Coordinate format: All entries are out-of-range.            |
   +-------------+-------------------------------------------------------------+
   | -5          | Matrix is singular and options.action=false                 |
   +-------------+-------------------------------------------------------------+
   | -6          | Matrix found not to be positive definite.                   |
   +-------------+-------------------------------------------------------------+
   | -7          | ptr[] and/or row[] not present, but required as             |
   |             | :c:func:`spldlt_analyse()` was called with check=false.     |
   +-------------+-------------------------------------------------------------+
   | -8          | options.ordering out of range, or options.ordering=0 and    |
   |             | order parameter not provided or not a valid permutation.    |
   +-------------+-------------------------------------------------------------+
   | -9          | options.ordering=-2 but val[] was not supplied.             |
   +-------------+-------------------------------------------------------------+
   | -10         | ldx<n or nrhs<1.                                            |
   +-------------+-------------------------------------------------------------+
   | -11         | job is out-of-range.                                        |
   +-------------+-------------------------------------------------------------+
   | -13         | Called :c:func:`spldlt_enquire_posdef()` on indefinite      |
   |             | factorization.                                              |
   +-------------+-------------------------------------------------------------+
   | -14         | Called :c:func:`spldlt_enquire_indef()` on                  |
   |             | positive-definite factorization.                            |
   +-------------+-------------------------------------------------------------+
   | -15         | options.scaling=3 but a matching-based ordering was not     |
   |             | performed during analyse phase.                             |
   +-------------+-------------------------------------------------------------+
   | -50         | Allocation error. If available, the stat parameter is       |
   |             | returned in inform.stat.                                    |
   +-------------+-------------------------------------------------------------+
   | -51         | CUDA error. The CUDA error return value is returned in      |
   |             | inform.cuda_error.                                          |
   +-------------+-------------------------------------------------------------+
   | -52         | CUBLAS error. The CUBLAS error return value is returned in  |
   |             | inform.cublas_error.                                        |
   +-------------+-------------------------------------------------------------+
   | +1          | Out-of-range variable indices found and ignored in input    |
   |             | data. inform.matrix_outrange is set to the number of such   |
   |             | entries.                                                    |
   +-------------+-------------------------------------------------------------+
   | +2          | Duplicate entries found and summed in input data.           |
   |             | inform.matrix_dup is set to the number of such entries.     |
   +-------------+-------------------------------------------------------------+
   | +3          | Combination of +1 and +2.                                   |
   +-------------+-------------------------------------------------------------+
   | +4          | One or more diagonal entries of :math:`A` are missing.      |
   +-------------+-------------------------------------------------------------+
   | +5          | Combination of +4 and +1 or +2.                             |
   +-------------+-------------------------------------------------------------+
   | +6          | Matrix is found be (structurally) singular during analyse   |
   |             | phase. This will overwrite any of the above warning flags.  |
   +-------------+-------------------------------------------------------------+
   | +7          | Matrix is found to be singular during factorize phase.      |
   +-------------+-------------------------------------------------------------+
   | +8          | Matching-based scaling found as side-effect of              |
   |             | matching-based ordering ignored                             |
   |             | (consider setting options.scaling=3).                       |
   +-------------+-------------------------------------------------------------+
   | +50         | OpenMP processor binding is disabled. Consider setting      |
   |             | the environment variable OMP_PROC_BIND=true (this may       |
   |             | affect performance on NUMA systems).                        |
   +-------------+-------------------------------------------------------------+

   .. c:member:: int cuda_error
   
      CUDA error code in the event of a CUDA error (0 otherwise).
      Note that due to asynchronous execution, CUDA errors may not be
      reported by the call that caused them.

   .. c:member:: int cublas_error
   
      cuBLAS error code in the event of a cuBLAS error (0 otherwise).

