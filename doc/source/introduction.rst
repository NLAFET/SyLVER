************
Introduction
************

=======
Purpose
=======

SyLVER is a sparse direct solver for computing the solution of **large
sparse symmetrically-structured linear systems** of equations. This
includes both positive-definite and indefinite sparse symmetric
systems as well as unsymmetric system whose sparsity pattern is
symmetric.

The solution of the system of equations:

.. math::

   AX = B

is achived by computing a factorization of the input matrix. 
the following cases are covered:

1. :math:`A` is **symmetric positive-definite**, we compute the
   **sparse Cholesky factorization**:

.. math::

   PAP^T = LL^T

where the factor :math:`L` is a lower triangular matrix and the matrix
:math:`P` is a permutation matrix used to reduce the `fill-in
<https://en.wikipedia.org/wiki/Sparse_matrix#Reducing_fill-in>`_
generated during the factorization. Following the matrix factorization
the solution can be retrieved by successively solving the system
:math:`LY=PB` (forward substitution) and :math:`L^{T}PX=Y` (backward
substitutions).

2. :math:`A` is **symmetric indefinite**, then we compute the
sparse :math:`LDL^T` decomposition:

.. math::

   A =  PLD(PL)^T

where :math:`P` is a permutation matrix, :math:`L` is unit lower triangular,
and :math:`D` is block diagonal with blocks of size :math:`1 \times 1`
and :math:`2 \times 2`.

3. :math:`A` is **unsymmetric**, then we compute the sparse :math
`LU` decomposition:

.. math::

   P_sAP_s^T = P_nLUQ_n

where :math:`P_s` is a permutation matrix corresponding to the
fill-reducing permutation whereas :math:`P_n` and :math:`Q_n` are
meant to improve the numerical property of the factorization
algorithm.  :math:`L` is lower triangular, and :math:`U` is unit upper
triangular.

The code optionally supports hybrid computation using one or more
NVIDIA GPUs.

SyLVER returns bit-compatible results.

An option exists to scale the matrix. In this case, the factorization
of the scaled matrix :math:`\overline{A} = S A S` is computed, where
:math:`S` is a diagonal scaling matrix.

==============
Usage overview
==============

Solving :math:`AX=B` using SyLVER is a four stage process.

- If :math:`A` is *symmetric*:

   1. Call :f:subr:`spldlt_analyse()` to perform a symbolic factorization, stored
      in `spldlt_akeep`.
   2. Call :f:subr:`spldlt_factor()` to perform a numeric
      factorization, stored in `spldlt_fkeep`. More than one numeric
      factorization can refer to the same `spldlt_akeep`.
   3. Call :f:subr:`spldlt__solve()` to perform a solve with the
      factors. More than one solve can be performed with the same
      `spldlt_fkeep`.
   4. Once all desired solutions have been performed, free memory with
      :f:subr:`spldlt_free()`.

- If :math:`A` is *unsymmetric*:

   1. Call :f:subr:`splu_analyse()` to perform a symbolic factorization, stored
      in `splu_akeep`.
   2. Call :f:subr:`splu_factor()` to perform a numeric
      factorization, stored in `splu_fkeep`. More than one numeric
      factorization can refer to the same `splu_akeep`.
   3. Call :f:subr:`splu_solve()` to perform a solve with the
      factors. More than one solve can be performed with the same
      `splu_fkeep`.
   4. Once all desired solutions have been performed, free memory with
      :f:subr:`splu_free()`.
         
.. note::

   The :f:subr:`sylver_init()` routine must be called before any other
   routines whitin SyLVER. When all the desired operations have been
   performed, the :f:subr:`sylver_finalize()` routine should be called.
