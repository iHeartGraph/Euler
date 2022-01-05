# The Euler interface

This folder contains the Euler interface, and two implementations of it: the detector and the predictor. 

The detector uses partially viewed graphs from time `t` to predict edges at time `t`. In practice, it would be used for forensic tasks, where one is performing an audit to identify anomalous connections that have already occurred.

The predictor uses fully viewd graphs from time `t` to predict edges at time `t+1`. In practice, it would be used as a live intrusion detection tool, as predictive models can score edges as they are observed--before they have been processed into full snapshots.