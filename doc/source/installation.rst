************
Installation
************

Quick Start
===========

Under Linux, or Mac OS X:

.. code-block:: bash

   # Get latest development version from github
   git clone https://github.com/NLAFET/sylver
   cd sylver

   mkdir build # create build directory
   cd build
   # configure compilation
   cmake <path-to-source> -D SYLVER_RUNTIME=StarPU -D SYLVER_ENABLE_CUDA=ON
   make # run compilation 

Third-party libraries
=====================

SPRAL
-----

`SPRAL <https://github.com/ralna/spral>`_ is an open-source library
for sparse linear algebra and associated algorithm and has several
important features used in SyLVER. By default, SPRAL is automatically
download and built during the installation of SyLVER. However, if you
wish to use your own version of SPRAL, which is not recommended, you
can use the instructions below to install it.

SPRAL installation
^^^^^^^^^^^^^^^^^^

The installation instruction presented here are only useful if you
wish to install SPRAL as an internal package. By default SPRAL is
automatically downloaded and built during the installation of SyLVER.

The latest release of SPRAL can be found on its `GitHub repository
<https://github.com/ralna/spral/releases>`_. The compilation of SPRAL
is handled by autotools and for example can be done as follow when
using the GCC compilers:

.. code-block:: bash

   cd spral
   mkdir build
   cd build
   ../configure CXX=g++ FC=gfortran CC=gcc CFLAGS="-g -O2 -march=native" CXXFLAGS="-g -O2 -march=native" FCFLAGS="-g -O2 -march=native" --with-metis="-L/path/to/metis -lmetis" --with-blas="-L/path/to/blas -lblas" --with-lapack="-L/path/to/lapack -llapack" --disable-openmp --disable-gpu
   make
   
Note that the compilation flags used for SPRAL **must match** the
flags used in the compilation of SyLVER. Here we use the flags ``-g -O2
-march=native`` that correspond to the ``RelWithDebInfo`` build type in
SyLVER.

Here we use the ``--disable-openmp`` option because SyLVER works with
the serial version of SPRAL. Additionally, in this example we disabled
the compilation of the SPRAL GPU kernels using the ``--disable-gpu``
option.

**Sequential version** of BLAS and LAPACK should be used. We recommend
using the `MKL <https://software.intel.com/mkl>`_ library for best
performance on Intel machines and `ESSL
<https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html>`_
on IBM machines. The `MKL link line advisor
<https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_
can be useful to fill the ``--with-blas`` and ``--with-lapack``
options.

When compiling SyLVER you need to provide both the path to the SPRAL
source directory which can be given using the ``-DSPRAL_SRC_DIR``
CMake option or the ``SPRAL_SRC_DIR`` environment variable and the
path to the SPRAL library which can be given using the ``-DSPRAL_DIR``
CMake option or the ``SPRAL_DIR`` environment variable.
                
METIS
-----
   
The `MeTiS <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_
partitioning library is needed by the SPRAL library and therefore,
needed when linking the SyLVER package for building examples and test
drivers.

When compiling SyLVER you can provide the path to the MeTiS library
using either ``-DMETIS_DIR`` CMake option or the ``METIS_DIR``
environment variable.

hwloc
-----

The `hwloc <https://www.open-mpi.org/projects/hwloc/>`_ library is
topology discovery library which is necessary for linking the examples
and test drivers if SPRAL was compiled with it. In this case, the
library path can be given to CMake using either the ``-D HWLOC_DIR``
definition or the ``HWLOC_DIR`` environment variable.

Runtime system
--------------

The ``-D SYLVER_RUNTIME=StarPU`` enables the compilation of the
parallel version of SyLVER using `StarPU runtime system
<http://starpu.gforge.inria.fr/>`_. In this case the StarPU version
needs to be at least 1.3.0. The StarPU library is found with the
``FindSTARPU.cmake`` script located in the ``cmake/Modules``
directory. Note that, for this script to be able to find the StarPU
library, you need to set the environment variable ``STARPU_DIR`` to
the path of you StarPU install base directory.

BLAS and LAPACK
---------------

The BLAS and LAPACK libraries play an important role in the
performance of the solver. We recommend using the `MKL
<https://software.intel.com/mkl>`_ library for best performance on
Intel machines and the `ESSL
<https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html>`_
library when running on IBM machines. Alternative BLAS and LAPACK
libraries include `OpenBLAS <https://www.openblas.net/>`_. Note that
SyLVER should be linked against the **sequential** BLAS and LAPACK
libraries.

These libraries are found via the CMake scripts `FindBLAS
<https://cmake.org/cmake/help/latest/module/FindBLAS.html>`_ and
`FindLAPACK
<https://cmake.org/cmake/help/latest/module/FindBLAS.html>`_ and
therefore it is possible to use the options ``-DBLA_VENDOR`` to
indicate which libraries to use. For example:

.. code-block:: bash

   cmake <path-to-source> -DBLA_VENDOR=Intel10_64lp_seq # configure compilation

selects and locates the sequential BLAS and LAPACK implementation for
the compilation and when linking test drivers, example and tests.

If CMake is unable to locate the requested libraries via the
``-D BLA_VENDOR``, it is still possible to give them explicitly using the
``-D LBLAS`` and ``-D LLAPACK`` options. For example:

.. code-block:: bash

   # configure compilation
   cmake <path-to-source> -D LBLAS="-L/path/to/blas -lblas" -D LLAPACK="-L/path/to/lapack -llapack"
