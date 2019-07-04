********
Examples
********

======
SpLDLT
======

Suppose we wish to factorize the matrix

.. math::

   A = \left(\begin{array}{ccccc}
      2. & 1.                \\
      1. & 4. & 1. &    & 1. \\
         & 1. & 3. & 2.      \\
         &    & 2. & -1.&    \\
         & 1. &    &    & 2.
   \end{array}\right)

and then solve for the right-hand side

.. math::

   B = \left(\begin{array}{c}
      4.    \\
      17.   \\
      19.   \\
      2.    \\
      12.
   \end{array}\right).

The following code may be used.

.. literalinclude:: ../../examples/spldlt_simple_example.F90
   :language: Fortran


This produces the following output::

    The computed solution is:
     1.0000000000E+00  2.0000000000E+00  3.0000000000E+00
     4.0000000000E+00  5.0000000000E+00
    Pivot order:   -3    4   -1    0   -2
