*****************
Basic Subroutines
*****************

In the below, all reals are double precision unless otherwise indicated.

======
SpLDLT
======

.. f:subroutine:: spldlt_analyse(akeep,n,ptr,row,options,inform, ncpu[,order,val])

   Perform the analyse (symbolic) phase of the factorization for a
   matrix supplied in `Compressed Sparse Column (CSC) format
   <http://www.numerical.rl.ac.uk/spral/doc/latest/Fortran/csc_format.html>`_. The
   resulting symbolic factors stored in `spldlt_akeep` should be passed
   unaltered in the subsequent calls to ssids_factor().

   :p spldlt_akeep akeep [out]: returns symbolic factorization, to be
      passed unchanged to subsequent routines.
   :p integer n [in]: number of columns in :math:`A`.
   :p integer(long) ptr(n+1) [in]: column pointers for :math:`A` (see
      `CSC format
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
   :p integer(long) ptr(n+1) [in]: column pointers for :math:`A` (see
      `CSC format
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
