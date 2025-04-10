# Federated Learning for Autoencoder-based Anomaly Detection in the Industrial IoT

This project investigates the use of autoencoders and federated learning for condition monitoring in Industrial IoT (IIoT) environments, with a focus on resource-constrained edge devices and data privacy. It was developed as part of a [Bachelor thesis](./Thesis.pdf) and published at [IEEE BigData 2022](https://ieeexplore.ieee.org/document/10020836).

## Architecture Overview

The project follows a stepwise approach:

1. **Autoencoder Optimization**  
   Development and tuning of a lightweight autoencoder-based anomaly detection model tailored to limited compute and memory environments.

2. **Federated Learning Integration**  
   Deployment of the model in a federated learning framework to ensure data privacy by exchanging model weights instead of raw data, thereby maintaining local data ownership while improving global generalizability.


 
Federated Learning IIoT Use Case Scenario                 | Federated Learning Training Cycle                                             
----------------------------------------------------------|-------------------------------------------------------------------------------
![Federated Learning](plots/ReadMe/federated-factory.png) | ![Federated Learning Architecture](plots/ReadMe/federated-training-cycle.png) 

---

## Evaluation & Results

To evaluate the success of this approach, we conducted a case study on a real-world industrial application of anomaly detection in rotating machines, which are commonly found in manufacturing.

Here, the performances and resource demands of three configurations were compared:

- A **baseline** model: centralized and resource-unconstrained
- A **centralized, resource-efficient** model: trained on pooled data  
- A **federated version** of the resource-efficient model: multiple instances trained locally on disjoint data subsets, exchanging only model weights

### Key Findings

Our research showed, that:
1. The proposed resource-efficient centralized model was able to achieve similar anomaly detection performance to the baseline architecture.
2. Even when used in a federated learning framework, only able to share model weights instead of data, instances of the resource-efficient model were still able to achive equal certainty of defect predictions.
3. At the same time, this approach succeeded in strongly improving resource consumption and guaranteeing data privacy, as no trainings data was ever required to leave individual devices.

![Resource evaluation](plots/ReadMe/E2-Resources-v2.png) 
![Transferlearning evaluation](plots/ReadMe/transferlearning-comparison.png)



## Project Structure & Usage

### Training & Models

- `src/training.py`: Training pipeline for the resource-efficient autoencoder  
- `src/baseline.py`: Baseline (resource-unconstrained) model  
- `src/worker_training.py`, `src/worker_aggregation.py`: Federated training and model aggregation logic  
- `config.yaml`: Central configuration for training parameters (e.g., LR, batch size, number of clients)


### Build & Deployment

- Dockerfiles for all components are located in `/docker`
- Supports local execution and deployment to cloud environments (e.g., Google Cloud Platform)
- Persistent state and model transfer mechanisms are built in for simulated or real federated setups
- Supports local execution and deployment to cloud environments (e.g., Google Cloud Platform) via ansible in `deployment/`
- For detailed information on deploying the KubeEdge testbed in GCP using Ansible, please refer to the [Deployment README](deployment/README.md).

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


