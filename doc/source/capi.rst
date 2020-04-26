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

   
.. c:function:: void spldlt_factorize(bool posdef, long const* ptr, int const* row, double const* val, double *scale, void *akeep, void **fkeep, sylver_options_t const* options, sylver_inform_t *inform)

.. c:function:: void spldlt_solve( int job, int nrhs, double *x, int ldx, void *akeep, void *fkeep, sylver_options_t const* options, sylver_inform_t *inform)

   Solve (for :math:`nrhs` right-hand sides) one of the following
   equations:
