- SOTA standard FL method
	- FedAvg
	- FedProx
	- SCAFFOLD
	- FedNova
	- MOON
	- FedDC
	- FedNTD
- # Type
	- Central Fedrated Learning: All edge devices communicate directly with the central server in a hub-and-spoke model.
	  logseq.order-list-type:: number
	- DistributedFedreated Learning
	  logseq.order-list-type:: number
		- **Hierarchical Structure**: Introduces multiple levels of aggregation with edge servers and cloud servers.
		- **Regional Aggregation**: Edge servers aggregate models from nearby devices before sending to higher-level servers.
		- **Reduced Latency**: Local aggregation reduces communication overhead to central servers.
		- **Partial Decentralization**: Still relies on some level of centralized coordination, but distributes the load.
		- **Use Case**: Better suited for geographically distributed deployments like mobile networks. #Mobile_Network
	- Swarm Learning
	  logseq.order-list-type:: number
		- **Fully Decentralized**: No central server; all nodes are peers with equal authority.
		- **Mesh Topology**: Nodes communicate peer-to-peer in a fully connected or partially connected mesh.
		- **Blockchain Coordination**: Uses distributed ledger technology for consensus and coordination without central authority.
		- **No Single Point of Failure**: The network continues functioning even if multiple nodes fail.
		- **Zero Trust Architecture**: No participant needs to trust a central entity; trust is distributed via blockchain.
- # Comparison
	- | Feature | Federated Learning | Distributed FL | Swarm Learning |
	  | ---- | ---- | ---- |
	  | **Coordination** | Central server | Hierarchical servers | Blockchain-based P2P |
	  | **Scalability** | Limited by server capacity | Better than FL | Highly scalable |
	  | **Fault Tolerance** | Low (single point) | Medium | High (resilient) |
	  | **Communication Overhead** | High (all to center) | Medium (hierarchical) | Variable (P2P) |
	  | **Privacy** | Server sees all updates | Partial aggregation | Maximum (distributed) |
	  | **Latency** | High for distant clients | Reduced via edge | Depends on network |
	  | **Governance** | Centralized | Semi-distributed | Fully distributed |
	  | **Implementation Complexity** | Low-Medium | Medium-High | High |
	  | **Trust Model** | Trust central server | Trust hierarchy | Trustless (blockchain) |
- # Federated Learning in B5G/6G
	- **1. Mobile Device Intelligence**
		- Smartphones train models for personalized services
		- Central operator aggregates for network-wide insights
		- Lower complexity than Swarm Learning
		- Suitable for operator-controlled scenarios
	- **2. Traffic Prediction**
		- Base stations report to regional controllers
		- Hierarchical aggregation for scalability
		- Centralized optimization of network resources
		- Faster convergence for time-critical applications
	- **3. Quality of Service (QoS) Optimization**
		- Distributed learning of QoS requirements
		- Central coordination for policy enforcement
		- Real-time adaptation to application needs
		- Simplified management
- # Distributed Federated Learning in B5G/6G
	- **1. Hierarchical Network Intelligence**
		- Cell-level learning → Regional aggregation → National optimization
		- Matches network architecture (RAN, edge, core)
		- Reduced latency compared to pure FL
		- Geographic optimization
	- **2. Multi-Access Edge Computing (MEC)**
		- Local edge servers aggregate from nearby devices
		- Reduced backhaul requirements
		- Regional model specialization
		- Better for latency-sensitive applications
	- **3. Cross-Tier Optimization**
		- Device tier → Edge tier → Cloud tier learning
		- Each tier specializes in appropriate timescales
		- Balanced between centralization and distribution
		- Practical for existing network hierarchies
- # Comparative Advantages for B5G/6G
	- ## When to Use Federated Learning
		- Single operator network
		- Centralized control is acceptable
		- Lower implementation complexity required
		- Faster convergence needed
		- Strong central governance
	- ## When to Use Distributed FL
		- Existing hierarchical network infrastructure
		- Geographic distribution important
		- Balance between centralization and distribution
		- Operators want regional control
		- Reducing backhaul traffic priority
	- ## When to Use Swarm Learning
		- Multi-operator collaboration required
		- Maximum privacy and autonomy needed
		- No trusted central authority available
		- Network resilience critical
		- Regulatory requirements prevent data centralization
		- Cross-border deployments
		- Competitive operators must collaborate
-
- # Paper Reference
	- [[Federated Learning in 5G and 6G Wireless Networks]]
	- [[Streamlined Secure P2P Federated Learning for 5G Defense]]