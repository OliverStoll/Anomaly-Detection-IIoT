# Ansible KubeEdge Deployment Playbook

This repository contains the Ansible playbooks and roles designed to provision a KubeEdge testbed for evaluation purposes on Google Cloud Platform (GCP). 
The playbooks automate the configuration and deployment of both Kubernetes worker nodes and edge nodes.

## Overview

This deployment module automates the following:
- Configuration of GCP credentials and SSH access.
- Provisioning of Kubernetes worker and edge node instances.
- Automated setup of the KubeEdge environment for testing and development purposes.

## Prerequisites

### GCP Credentials and SSH Key
1. **GCP Service Account Credential:**  
   Create a `.google-cred.json` file based on the instructions provided in the [GCP Credential Guide](https://docs.ansible.com/ansible/latest/scenario_guides/guide_gce.html).  
   Place this file in the same directory as the playbooks, or update the file path in the configuration accordingly.

2. **SSH Key:**  
   Place the private SSH key used for GCP access as `gcp-key` in the playbook directory. Alternatively, modify the `private_key_file` path in the configuration file if the key is stored elsewhere.

3. **User Configuration:**  
   Adjust the `remote_user` setting in the Ansible configuration (`config/config.cfg`) to reflect the appropriate username for remote access.

4. **GCP Project Identification:**  
   Update the `gcp_project` variable in both `scenario.yml` and `inventory.gcp.yml` with the correct GCP project ID.

### Ansible Environment
Ensure that Ansible and related dependencies are installed within an isolated Python environment:

```bash
# Create a virtual environment using Python 3.8 (or later)
virtualenv -p python3.8 .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Ansible Galaxy roles and collections as defined in requirements.yml
ansible-galaxy install -r requirements.yml
```

#### Bootstrap testbed
1) Adjust `scenario.yml` and specify the required kubernetes worker and edge node instances.

2) Run the playbook:
```
export ANSIBLE_CONFIG=deployment/config/config.cfg

ansible-playbook \
  -i deployment/inventory/inventory.gcp.yml \
  deployment/playbooks/bootstrap-kubeedge-gcp.yml \
  -e "@deployment/playbooks/scenario.yml"
``` 
