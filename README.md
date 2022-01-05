# Euler: Detecting Network Lateral Movement via Scalable Temporal Graph Link Prediction

## Abstract
Lateral movement is a key stage of system compromise used by advanced persistent threats. Detecting it is no simple task. When network host logs are abstracted into discrete temporal graphs, the problem can be reframed as anomalous edge detection in an evolving network. Research in modern deep graph learning techniques has produced many creative and complicated models for this task. However, as is the case in many machine learning fields, generality of models is of paramount importance to achieve good accuracy and scalability in training and inference.  
	
In this paper, we propose a formalized version of this approach in a framework we call Euler. It consists of a model-agnostic graph neural network stacked upon a model-agnostic sequence encoding layer such as a recurrent neural network. Models built according to the Euler framework can easily distribute their graph convolutional layers across multiple machines for large performance improvements. We demonstrate that Euler-based models are competitive, or better than many state-of-the-art approaches to anomalous link detection and prediction. As anomaly-based intrusion detection systems, Euler models can efficiently identify anomalous connections between entities with high precision and outperform other unsupervised techniques for anomalous lateral movement detection. 

## Motivation
Most temporal link prediction models which utilize graph neural networks do not scale well. Almost every model in this field uses a GNN combined with an RNN in such a way where the GNN input is dependant on the RNN output, as shown below: 

![](/img/sota.png)

Our framework separates the two such that GNNs are free to run independantly. In this way, they can be distributed across multiple machines for large performance improvements

![](/img/model.png)

This framework is scalable to large datasets, and is much faster than every other temporal link prediction method we tried, while retaining similar, or better precision and AUC. 

<img src="/img/scalability.png" width="375"/>) <img src="/img/runtimes.png" width="350"/>
