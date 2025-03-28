.. Node Vector Distance documentation master file, created by
   sphinx-quickstart on Fri Mar 28 10:20:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Node Vector Distance documentation!
==================================================

``Node Vector Distance`` (or ``NVD``) is a Python module to calculate efficiently several node attribute measures.

The library currently provides unique methods to:

1. Calculate Euclidean distances [generalized-euclidean]_ and correlations [network-pearson]_ between pairs of node attributes over the graph, or the variance [graph-variance]_ of a node attribute.
2. Generate efficiently useful graph matrix representations such as the `pseudoinverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ of the `Laplacian <https://en.wikipedia.org/wiki/Laplacian_matrix>`_ and the `effective resistance <https://en.wikipedia.org/wiki/Resistance_distance>`_ matrix.
3. Integrate with `torch <https://pytorch.org/>`_ and `torch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ which can lead to perform the above tasks on the GPU, orders of magnitude faster than using standard CPU solutions.

Installing NVD
---------------------
.. code-block:: bash

   pip install node_vector_distance

Note, however, that in order to use the GPU capabilities of the library you will need to have a valid installation of torch and torch-geometric. It might be worht having these libraries installed even without GPU integration, as they can sometimes provide faster options even when running on the CPU. As long as those two packages are properly installed, you don't have to have them run on the GPU to use ``NVD``.

Getting started
---------------

You should read first the :ref:`quick start guide <quickstart>`

Reporting bugs / Proposing features
-----------------------------------

If you encounter a problem, or have a suggestion for a possible improvement, open an issue in the `git repository <https://github.com/mikk-c/node_vector_distance/issues>`_.

References
----------
.. [generalized-euclidean] Coscia, Michele. "Generalized Euclidean measure to estimate network distances." In Proceedings of the international AAAI conference on web and social media, vol. 14, pp. 119-129. 2020. :doi:10.1609/icwsm.v14i1.7284 
.. [network-pearson] Coscia, Michele. "Pearson correlations on complex networks." Journal of Complex Networks 9, no. 6 (2021): cnab036. :doi:10.1093/comnet/cnab036
.. [graph-variance] Devriendt, Karel, Samuel Martin-Gutierrez, and Renaud Lambiotte. "Variance and covariance of distributions on graphs." SIAM Review 64, no. 2 (2022): 343-359. :doi:10.1137/20M1361328

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   quickstart