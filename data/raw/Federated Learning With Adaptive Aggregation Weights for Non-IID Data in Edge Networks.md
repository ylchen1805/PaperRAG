# Data heterogeneity
	- non-IID data
	- Arise fron differences in environments, data sources, user perferemces, amd hardware across edge devices.
	- ## Impact
		- Performance degradation and Non-Convergence
			- Experimental results show that as the degree of heterogeneity increases—such as when the Dirichlet parameter *α* decreases—standard methods like FedAvg can see accuracy plummet
		- Local objective drift
			- When data is heterogeneous, each device tends to adapt its local model parameters to its ==own specific data distribution==
			- Local models toward the ==optima of local objectivesrather== than the global optimum
			- The global model is often poor because it is a combination of models that have drifted in different, conflicting direction
		- Suboptimality of standard aggregation weight
			- The dataset size does not adequately reflect non-IID properties, so the aggregation based on the proportion of dataset sizes ==is not optimal strategy==
			- A tighter convergence bound can be achieved by considering the ==squared full-batch gradient norm== of each device, which captures the importance and fit of the local model to the training process #Squared_full_batch_gradient_norm #Convergence_bound
		- Impact on communicatioin and stability
			- Training process is ==unstable and less efficient==
			- Methods that work well under uniform( IID) data distributions often become ==unstable or **fail**== when data is highly partitioned across devices.
			- This necessitates more complex regularization or adaptive weighting strategies, such as ==FedAAW==, to mitigate the drift and restore performance #Adaptive_weighting
	- #Data_heterogeneity
- # Federated Learning with Adaptive Aggregation Weights( FedAAW)
- Minimizing convergence bounds
  logseq.order-list-type:: number
	- A tighter convergence bound is achieved when the aggregation weights are optimized based not only on local dataset size but also on the ==squared full-batch gradient== norm of each device.
- Squared full-batch gradient norm
  logseq.order-list-type:: number
	- A **smaller** squared gradient norm indicates that the local model already fits its data well.
	- $$\lvert \lvert \nabla F_k(w^t_{k,0})\rvert\rvert ^ 2$$
	- Stand for the ==importance or fit== of a local model
	- Assigns **higher weights** to devices with smaller squared gradient norms because these models are deemed to carry more stable and beneficial information for the global training proccesss.
- The Adaptive weighting mechanism
  logseq.order-list-type:: number
	- Numerical Tracker $R^t_k$
		- The average of the cumulative squared full-batch gradient nonrms for device $k$ from all past rounds to round $t$
- #Aggregation