# 🤖 Federated Learning for Autoencoder-based Anomaly Detection in the Industrial IoT

This project investigates the use of autoencoders and federated learning for condition monitoring in Industrial IoT (IIoT) environments, with a focus on resource-constrained edge devices and data privacy. It was developed as part of my [Bachelor thesis](./Thesis.pdf) and published at [IEEE BigData 2022](https://ieeexplore.ieee.org/document/10020836).

## 🏗️ System Architecture

The project follows a stepwise approach:

#### ⚙️ Lightweight Autoencoder Design
A compact and efficient autoencoder was developed for detecting anomalies in sensor data from industrial machinery. The model is optimized to run on edge devices with constrained compute and memory capacity.

#### 🔐 Federated Learning for Privacy
To preserve data privacy, the system leverages a federated learning setup. Instead of transferring raw data, only model parameters are shared among devices. 
This allows local data to remain on-premise while enabling global model improvements through collaboration.


 
Federated Learning IIoT Use Case Scenario                 | Federated Learning Training Cycle                                             
----------------------------------------------------------|-------------------------------------------------------------------------------
![Federated Learning](plots/ReadMe/federated-factory.png) | ![Federated Learning Architecture](plots/ReadMe/federated-training-cycle.png) 

---

## 📊 Evaluation & Results

To evaluate the success of this approach, we conducted a case study on a real-world industrial application of anomaly detection in rotating machines, which are commonly found in manufacturing.

Here, the performances and resource demands of three configurations were compared:

- A **baseline** model: centralized and resource-unconstrained
- A **centralized, resource-efficient** model: trained on pooled data  
- A **federated version** of the resource-efficient model: multiple instances trained locally on disjoint data subsets, exchanging only model weights

### 🔍 Key Findings

Our research showed, that:
1. The proposed resource-efficient centralized model was able to achieve similar anomaly detection performance to the baseline architecture.
2. Even when used in a federated learning framework, only able to share model weights instead of data, instances of the resource-efficient model were still able to achive equal certainty of defect predictions.
3. At the same time, this approach succeeded in strongly improving resource consumption and guaranteeing data privacy, as no trainings data was ever required to leave individual devices.

![Resource evaluation](plots/ReadMe/E2-Resources-v2.png) 
![Transferlearning evaluation](plots/ReadMe/transferlearning-comparison.png)



## 🧱 Project Structure & Usage

#### 🧪  Training & Models

- `src/models/`: Resource-efficient condition monitoring model for deployment at the edge
- `src/training/`: Training pipeline for the resource-efficient autoencoder, as well as baseline (resource-unconstrained) condition monitoring model for comparison
- `src/federated_learning/`: Federated training, communication between models and model aggregation logic
- `src/data/`: Entire data pipeline for data loading, cleaning and transformation 
- `config.yaml`: Central configuration for training parameters (e.g., LR, batch size, number of clients)


#### 🚀 Build & Deployment

- Dockerfiles for all components are located in `docker/`
- Persistent state and model transfer mechanisms are built in for simulated or real federated setups
- Supports local execution and deployment to cloud environments (e.g., Google Cloud Platform) via ansible in `deployment/`
- For detailed information on deploying the KubeEdge testbed in GCP using Ansible, please refer to the [Deployment README](deployment/README.md).

---

## 📄 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


