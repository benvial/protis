protis
======

.. py:module:: protis

.. autoapi-nested-parse::

   This module implements the protis API.



Classes
-------

.. autoapisummary::

   protis.Simulation
   protis.VersionTable
   protis.VersionTable


Functions
---------

.. autoapisummary::

   protis.set_backend
   protis.use_gpu
   protis.print_info
   protis.init_bands
   protis.init_bands_plot
   protis.plot_bands
   protis.gen_eig
   protis.gram_schmidt
   protis.plot2d
   protis.is_scalar
   protis.block_anisotropic
   protis.block_z_anisotropic
   protis.block
   protis.is_z_anisotropic
   protis.is_symmetric
   protis.is_hermitian
   protis.invblock
   protis.is_scalar
   protis.block_anisotropic
   protis.block_z_anisotropic
   protis.block
   protis.is_z_anisotropic
   protis.is_symmetric
   protis.is_hermitian
   protis.invblock


Package Contents
----------------

.. py:function:: set_backend(backend)

   Set the numerical backend used by protis.

   :param backend: The backend to use. Must be one of "numpy", "scipy", "autograd", "jax" or "torch".
   :type backend: str

   .. rubric:: Notes

   This function is a wrapper around nannos.set_backend and also reloads the protis package.


.. py:function:: use_gpu(boolean)

   Enable or disable GPU usage for computations.

   :param boolean: If True, set the system to use GPU for computations; if False, use CPU.
   :type boolean: bool

   .. rubric:: Notes

   This function sets the GPU usage state for the current session and reloads
   the package to apply the changes.


.. py:function:: print_info()

.. py:function:: init_bands(sym_points, nband)

.. py:function:: init_bands_plot(sym_points, nband)

.. py:function:: plot_bands(sym_points, nband, eigenvalues, xtickslabels=None, color=None, **kwargs)

.. py:function:: gen_eig(A, B, vectors=True, sparse=False, neig=10, **kwargs)

.. py:function:: gram_schmidt(A, norm=True, row_vect=False)

   Orthonormalizes vectors by gram-schmidt process

   :param A:
   :type A: ndarray,
   :param Matrix having vectors in its columns:
   :param norm:
   :type norm: bool,
   :param Do you need Normalized vectors?:
   :param row_vect:
   :type row_vect: bool,
   :param Does Matrix A has vectors in its rows?:

   :returns: * **G** (*ndarray,*)
             * *Matrix of orthogonal vectors*


.. py:class:: Simulation(lattice, k=(0, 0), epsilon=1, mu=1, nh=100)

   .. py:attribute:: lattice


   .. py:attribute:: k
      :value: (0, 0)



   .. py:attribute:: epsilon
      :value: 1



   .. py:attribute:: mu
      :value: 1



   .. py:attribute:: nh0
      :value: 100



   .. py:property:: kx


   .. py:property:: ky


   .. py:property:: Kx


   .. py:property:: Ky


   .. py:method:: build_epsilon_hat()


   .. py:method:: build_mu_hat()


   .. py:method:: build_A(polarization)


   .. py:method:: build_B(polarization)


   .. py:method:: solve(polarization, vectors=True, rbme=None, return_square=False, sparse=False, neig=10, ktol=1e-12, reduced=False, **kwargs)


   .. py:method:: get_rbme_matrix(N_RBME, bands_RBME, polarization)


   .. py:method:: unit_cell_integ(u)


   .. py:method:: phasor()


   .. py:method:: get_chi(polarization)


   .. py:method:: get_xi(polarization)


   .. py:method:: build_Cs(phi0, polarization)


   .. py:method:: normalization(mode, polarization)


   .. py:method:: coeff2mode(coeff)


   .. py:method:: get_mode(imode)


   .. py:method:: get_modes(imodes)


   .. py:method:: scalar_product_real(u, v)


   .. py:method:: scalar_product_fourier(u, v)


   .. py:method:: scalar_product_rbme(u, v)


   .. py:method:: plot(field, nper=1, ax=None, cmap='RdBu_r', show_cell=False, cellstyle='-k', **kwargs)


   .. py:method:: get_hfh_tensor(imode, polarization)


   .. py:method:: get_berry_curvature(kx, ky, eigenmode, method='fourier')


   .. py:method:: get_chern_number(kx, ky, berry_curvature)


.. py:function:: plot2d(lattice, grid, field, nper=1, ax=None, cmap='RdBu_r', show_cell=False, cellstyle='-k', **kwargs)

.. py:class:: VersionTable(shell=None, **kwargs)

   Bases: :py:obj:`IPython.core.magic.Magics`


   A class of status magic functions.


   .. py:method:: protis_version_table(line='', cell=None)

      Print an HTML-formatted table with version numbers for Protis and its
      dependencies. This should make it possible to reproduce the environment
      and the calculation later on.



.. py:function:: is_scalar(a)

.. py:function:: block_anisotropic(a, dim=3)

.. py:function:: block_z_anisotropic(axx, axy, ayx, ayy, azz)

.. py:function:: block(a)

.. py:function:: is_z_anisotropic(a)

.. py:function:: is_symmetric(M)

.. py:function:: is_hermitian(M)

.. py:function:: invblock(A, B, C, D)

.. py:class:: VersionTable(shell=None, **kwargs)

   Bases: :py:obj:`IPython.core.magic.Magics`


   A class of status magic functions.


   .. py:method:: protis_version_table(line='', cell=None)

      Print an HTML-formatted table with version numbers for Protis and its
      dependencies. This should make it possible to reproduce the environment
      and the calculation later on.



.. py:function:: is_scalar(a)

.. py:function:: block_anisotropic(a, dim=3)

.. py:function:: block_z_anisotropic(axx, axy, ayx, ayy, azz)

.. py:function:: block(a)

.. py:function:: is_z_anisotropic(a)

.. py:function:: is_symmetric(M)

.. py:function:: is_hermitian(M)

.. py:function:: invblock(A, B, C, D)

