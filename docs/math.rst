.. _math:

Mathematical Foundations
========================

This chapter explains the ideas and numerical principles that make GAMs work —
not just the equations, but *why* we chose each ingredient and what problem it solves.

gamkit provides two distinct engines that solve the same penalized regression problem with
different statistical machinery. The mathematical foundations are organized into four
self-contained chapters:

.. toctree::
   :maxdepth: 2

   math/foundations
   math/eilers
   math/wood
   math/comparison

Overview & Chapter Guide
------------------------

* :doc:`math/foundations` (:ref:`Part I <section-why-gam>`):
  The shared ideas both backends agree on: what a GAM is, why additive models balance
  flexibility with interpretability, and how predictions decompose into independent component plots.

* :doc:`math/eilers` (:ref:`Part II <section-eilers-design>`):
  The **Eilers-Marx backend**: uniform B-splines, discrete difference penalties, SVD data-augmented
  solver, Generalized Cross Validation (GCV), fast QR+Cholesky grid search, and observation weights.

* :doc:`math/wood` (:ref:`Part III <section-wood-design>`):
  The **Wood backend**: Simon Wood's rich basis families (thin plate, cubic, cyclic, P-splines, random effects),
  null-space constraint absorption, sum-to-zero identifiability, and continuous REML/GCV optimization
  with approximate F-tests and p-values (Wood 2013).

* :doc:`math/comparison` (:ref:`Part IV <section-backends-compared>`):
  An explicit side-by-side comparison of the two backends (feature matrix, numerical methods, scaling,
  and decision rubric) so you can choose the optimal engine for your data.

Where to Start?
---------------

* If you're new to GAMs, start with :doc:`math/foundations`.
* If you already understand penalized splines and want implementation details, skip to
  :ref:`the SVD solver <section-svd-solver>` (Eilers-Marx) or :ref:`the Wood bases <section-wood-bases>`.
* If you need help choosing between the two engines, jump straight to :ref:`the comparison & selection guide <section-backends-compared>`.
