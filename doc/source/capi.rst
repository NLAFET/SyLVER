*****
C API
*****

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
