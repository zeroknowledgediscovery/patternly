===============
patternly
===============

.. image:: http://zed.uchicago.edu/logo/patternly.png
   :height: 200px
   :alt: patternly logo
   :align: center

.. class:: no-web no-pdf

:Info: Paper draft link will be posted here
:Author: Drew Vlasnik, Ishanu Chattopadhyay
:Laboratory: The Laboratory for Zero Knowledge Discovery, The University of Chicago  https://zed.uchicago.edu
:Description: Discovery of emergent anomalies in data streams without explicit  prior models of correct or aberrant behavior, based on the modeling of ergodic, quasi-stationary finite valued processes as probabilistic finite state automata (PFSA_).
:Documentation: https://zeroknowledgediscovery.github.io/patternly

.. _PFSA: https://pubmed.ncbi.nlm.nih.gov/23277601/


**Installation:**

.. code-block::

    pip install patternly --user -U


**Usage:**

See `examples`_.

.. _examples: https://github.com/zeroknowledgediscovery/patternly/tree/main/examples

.. code-block::

    from patternly.detection import AnomalyDetection, StreamingDetection

