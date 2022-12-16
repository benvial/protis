
.. |release_badge| image:: https://img.shields.io/endpoint?url=https://gitlab.com/protis/protis/-/jobs/artifacts/main/raw/logobadge.json?job=badge
  :target: https://gitlab.com/protis/protis/-/releases
  :alt: Release

.. |GL_CI| image:: https://img.shields.io/gitlab/pipeline/protis/protis/main?logo=gitlab&labelColor=grey&style=for-the-badge
  :target: https://gitlab.com/protis/protis/commits/main
  :alt: pipeline status

.. .. |conda| image:: https://img.shields.io/conda/vn/conda-forge/protis?logo=conda-forge&color=CD5C5C&logoColor=white&style=for-the-badge   
..   :target: https://anaconda.org/conda-forge/protis
..   :alt: Conda (channel only)

.. .. |conda_dl| image:: https://img.shields.io/conda/dn/conda-forge/protis?logo=conda-forge&logoColor=white&style=for-the-badge
..   :alt: Conda

.. .. |conda_platform| image:: https://img.shields.io/conda/pn/conda-forge/protis?logo=conda-forge&logoColor=white&style=for-the-badge
..   :alt: Conda


.. |pip| image:: https://img.shields.io/pypi/v/protis?color=blue&logo=pypi&logoColor=e9d672&style=for-the-badge
  :target: https://pypi.org/project/protis/
  :alt: PyPI
  
.. |pip_dl| image:: https://img.shields.io/pypi/dm/protis?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Downloads
   
.. |pip_status| image:: https://img.shields.io/pypi/status/protis?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=e9d672&style=for-the-badge
  :alt: Code style: black
 
.. |coverage| image:: https://img.shields.io/gitlab/coverage/protis/protis/main?logo=python&logoColor=e9d672&style=for-the-badge
  :target: https://gitlab.com/protis/protis/commits/main
  :alt: coverage report 
  
.. |zenodo| image:: https://img.shields.io/badge/DOI-10.5281/zenodo.6636140-dd7d54?logo=google-scholar&logoColor=dd7d54&style=for-the-badge
  :target: https://doi.org/10.5281/zenodo.6636140
 
.. |licence| image:: https://img.shields.io/badge/license-GPLv3-blue?color=dd7d54&logo=open-access&logoColor=dd7d54&style=for-the-badge
  :target: https://gitlab.com/protis/protis/-/blob/main/LICENCE.txt
  :alt: license

+----------------------+----------------------+----------------------+
| Release              |            |release_badge|                  |
+----------------------+----------------------+----------------------+
| Deployment           |                   |pip|                     |
+----------------------+----------------------+----------------------+
| Build Status         |            |GL_CI|                          |
+----------------------+----------------------+----------------------+
| Metrics              |                |coverage|                   |
+----------------------+----------------------+----------------------+
| Activity             |                  |pip_dl|                   |
+----------------------+----------------------+----------------------+
| Citation             |           |zenodo|                          |
+----------------------+----------------------+----------------------+
| License              |           |licence|                         |
+----------------------+----------------------+----------------------+
| Formatter            |           |black|                           |
+----------------------+----------------------+----------------------+



.. inclusion-marker-badges

=============================================================
protis: Plane wave expansion method for photonic crystals
=============================================================


.. inclusion-marker-install-start

Installation
============

From pypi
---------

The package is available on `pypi <https://pypi.org/project/protis>`_.
To install, simply use:

.. code-block:: bash

  pip install protis


From sources
-------------

Sources are available on `gitlab <https://gitlab.com/protis/protis>`_. First
clone the repository and install with ``pip``:

.. code-block:: bash

  git clone https://gitlab.com/protis/protis.git
  cd protis
  pip install -e .


.. inclusion-marker-install-end


Documentation
=============

The reference documentation and examples can be found on the
`project website <https://protis.gitlab.io>`_.


License
=======


.. inclusion-marker-license-start

This software is published under the `GPLv3 license <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.


.. inclusion-marker-license-end
