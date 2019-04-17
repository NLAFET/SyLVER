**********
Data types
**********

=======
Options
=======

.. f:type:: sylver_options

   The derived data type :f:type:`sylver_options` is used to specify the
   options used within ``SyLVER``. The components, that are
   automatically given default values in the definition of the type,
   are:

   :f integer print_level [default=0]: the level of printing. The different 
      levels are:

      +----------+-------------------------------------------------+
      | < 0      | No printing.                                    |
      +----------+-------------------------------------------------+
      | = 0      | Error and warning messages only.                |
      +----------+-------------------------------------------------+
      | = 1      | As 0, plus basic diagnostic printing.           |
      +----------+-------------------------------------------------+
      | > 1      | As 1, plus some additional diagnostic printing. |
      +----------+-------------------------------------------------+

   :f integer unit_diagnostics [default=6]: Fortran unit number for 
      diagnostics printing. Printing is suppressed if <0.

   :f integer unit_error [default=6]: Fortran unit number for printing of
      error messages. Printing is suppressed if <0.
   :f integer unit_warning [default=6]: Fortran unit number for printing of
      warning messages. Printing is suppressed if <0.
   :f integer ordering [default=1]: Ordering method to use in analyse phase:

      +-------------+---------------------------------------------------------+
      | 0           | User-supplied ordering is used (`order` argument to     |
      |             | :f:subr:`spldlt_analyse()` or                           |
      |             | :f:subr:`splu_analyse()`).                              |
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
      |             | be used in :f:subr:`spldlt_factor()` or                 |
      |             | :f:subr:`splu_factor()` (see %scaling below).           |
      +-------------+---------------------------------------------------------+

   :f integer nemin [default=32]: supernode amalgamation threshold. Two
      neighbours in the elimination tree are merged if they both involve fewer
      than nemin eliminations. The default is used if nemin<1.
   :f logical use_gpu [default=true]: Use an NVIDIA GPU if present.
   :f integer scaling [default=0]: scaling algorithm to use:

      +---------------+-------------------------------------------------------+
      | <=0 (default) | No scaling (if ``scale(:)`` is not present on call to |
      |               | :f:subr:`spldlt_factor()` or :f:subr:`splu_factor()`, |
      |               | or user-supplied scaling (if ``scale(:)`` is present).|
      +---------------+-------------------------------------------------------+
      | =1            | Compute using weighted bipartite matching via the     |
      |               | Hungarian Algorithm (``MC64`` algorithm).             |
      +---------------+-------------------------------------------------------+
      | =2            | Compute using a weighted bipartite matching via the   |
      |               | Auction Algorithm (may be lower quality than that     |
      |               | computed using the Hungarian Algorithm, but can be    |
      |               | considerably faster).                                 |
      +---------------+-------------------------------------------------------+
      | =3            | Use matching-based ordering generated during the      |
      |               | analyse phase using options%ordering=2. The scaling   |
      |               | will be the same as that generated with               |
      |               | options%scaling= 1 if the matrix values have not      |
      |               | changed. This option will generate an error if a      |
      |               | matching-based ordering was not used during analysis. |
      +---------------+-------------------------------------------------------+
      | >=4           | Compute using the norm-equilibration algorithm of     |
      |               | Ruiz.                                                 |
      +---------------+-------------------------------------------------------+

   :f integer nb [default=256]: Block size to use for
      parallelization of large nodes on CPU resources.

   :f integer pivot_method [default=1]: Pivot method to be used on CPU, one of:

      +-------------+----------------------------------------------------------+
      | 0           | Aggressive a posteori pivoting. Cholesky-like            |
      |             | communication pattern is used, but a single failed pivot |
      |             | requires restart of node factorization and potential     |
      |             | recalculation of all uneliminated entries.               |
      +-------------+----------------------------------------------------------+
      | 1 (default) | Block a posteori pivoting. A failed pivot only requires  |
      |             | recalculation of entries within its own block column.    |
      +-------------+----------------------------------------------------------+
      | 2           | Threshold partial pivoting. Not parallel.                |
      +-------------+----------------------------------------------------------+

   :f real small [default=1d-20]: threshold below which an entry is treated as
      equivalent to `0.0`.
   :f real u [default=0.01]: relative pivot threshold used in symmetric
      indefinite case. Values outside of the range :math:`[0,0.5]` are treated
      as the closest value in that range.

===========
Information
===========

.. f:type:: sylver_inform

   The derived data type :f:type:`sylver_inform` is used to return
   information about the progress and needs of the algorithm that
   might be of interest for the user.

   :f integer flag: exit status of the algorithm (see table below).
   :f integer cublas_error: CUBLAS error code in the event of a CUBLAS error
      (0 otherwise).
   :f integer cuda_error: CUDA error code in the event of a CUDA error
      (0 otherwise). Note that due to asynchronous execution, CUDA errors may 
      not be reported by the call that caused them.
   :f integer matrix_dup: number of duplicate entries encountered (if
      :f:subr:`spldlt_analyse()` or :f:subr:`splu_analyse()` called with
      check=true).
   :f integer matrix_missing_diag: number of diagonal entries without
      an explicit value (if :f:subr:`spldlt_analyse()` or
      :f:subr:`splu_analyse()` called with check=true).
   :f integer matrix_outrange: number of out-of-range entries
      encountered (if :f:subr:`spldlt_analyse()` or
      :f:subr:`splu_analyse()` called with check=true).
   :f integer matrix_rank: (estimated) rank (structural after analyse
      phase, numerical after factorize phase).
   :f integer maxdepth: maximum depth of the assembly tree.
   :f integer maxfront: maximum front size (without pivoting after
      analyse phase, with pivoting after factorize phase).
   :f integer num_delay: number of delayed pivots. That is, the total
      number of fully-summed variables that were passed to the father node
      because of stability considerations. If a variable is passed further
      up the tree, it will be counted again.
   :f long num_factor: number of entries in :math:`L`
      (without pivoting after analyse phase, with pivoting after
      factorize phase).
   :f long num_flops: number of
      floating-point operations for Cholesky factorization (indefinte
      needs slightly more). Without pivoting after analyse phase, with
      pivoting after factorize phase.
   :f integer num_neg: number of negative eigenvalues of the matrix
      :math:`D` after factorize phase.
   :f integer num_sup: number of supernodes in assembly tree.
   :f integer num_two: number of :math:`2 \times 2` pivots used by the
      factorization (i.e. in the matrix :math:`D` in the indefinite
                       case).
   :f integer stat: Fortran allocation status parameter in event of
      allocation error (0 otherwise).

   +-------------+-------------------------------------------------------------+
   | inform%flag | Return status                                               |
   +=============+=============================================================+
   | 0           | Success.                                                    |
   +-------------+-------------------------------------------------------------+
   | -1          | Error in sequence of calls (may be caused by failure of a   |
   |             | preceding call).                                            |
   +-------------+-------------------------------------------------------------+
   | -2          | n<0 or ne<1.                                                |
   +-------------+-------------------------------------------------------------+
   | -3          | Error in ptr(:).                                            |
   +-------------+-------------------------------------------------------------+
   | -4          | CSC format: All variable indices in one or more columns are |
   |             | out-of-range.                                               |
   |             |                                                             |
   |             | Coordinate format: All entries are out-of-range.            |
   +-------------+-------------------------------------------------------------+
   | -5          | Matrix is singular and options%action=.false.               |
   +-------------+-------------------------------------------------------------+
   | -6          | Matrix found not to be positive definite but posdef=true.   |
   +-------------+-------------------------------------------------------------+
   | -7          | ptr(:) and/or row(:) not present although required.         |
   +-------------+-------------------------------------------------------------+
   | -8          | options%ordering out of range, or options%ordering=0 and    |
   |             | order parameter not provided or not a valid permutation.    |
   +-------------+-------------------------------------------------------------+
   | -9          | options%ordering=-2 but val(:) was not supplied.            |
   +-------------+-------------------------------------------------------------+
   | -10         | ldx<n or nrhs<1.                                            |
   +-------------+-------------------------------------------------------------+
   | -11         | job is out-of-range.                                        |
   +-------------+-------------------------------------------------------------+
   | -13         | Called :f:subr:`ssids_enquire_posdef()` on indefinite       |
   |             | factorization.                                              |
   +-------------+-------------------------------------------------------------+
   | -14         | Called :f:subr:`ssids_enquire_indef()` on positive-definite |
   |             | factorization.                                              |
   +-------------+-------------------------------------------------------------+
   | -15         | options%scaling=3 but a matching-based ordering was not     |
   |             | performed during analyse phase.                             |
   +-------------+-------------------------------------------------------------+
   | -50         | Allocation error. If available, the stat parameter is       |
   |             | returned in inform%stat.                                    |
   +-------------+-------------------------------------------------------------+
   | -51         | CUDA error. The CUDA error return value is returned in      |
   |             | inform%cuda_error.                                          |
   +-------------+-------------------------------------------------------------+
   | -52         | CUBLAS error. The CUBLAS error return value is returned in  |
   |             | inform%cublas_error.                                        |
   +-------------+-------------------------------------------------------------+
   | +1          | Out-of-range variable indices found and ignored in input    |
   |             | data. inform%matrix_outrange is set to the number of such   |
   |             | entries.                                                    |
   +-------------+-------------------------------------------------------------+
   | +2          | Duplicate entries found and summed in input data.           |
   |             | inform%matrix_dup is set to the number of such entries.     |
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
   |             | (consider setting options%scaling=3).                       |
   +-------------+-------------------------------------------------------------+

