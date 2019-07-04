**************
Usage overview
**************

For Solving :math:`AX=B` using SyLVER is a four stage process.

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
