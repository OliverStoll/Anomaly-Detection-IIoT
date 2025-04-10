# Ansible Kubeedge Deployment Playbook
Ansible playbooks and roles to create a KubeEdge testbed in GCP.


## Dependencies

#### GCP credentials
1) Create a `.google-cred.json` file in the playbook dir following [this guide](https://docs.ansible.com/ansible/latest/scenario_guides/guide_gce.html).

2) Place the private SSH key used in your GCP project as `gcp-key` in the playbook dir, or adjust the path in the following config files.

3) Adjust the `remote_user` variable in `ansible.cfg`.

4) Adjust `gcp_project` in `scenario.yml` and `inventory.gcp.yml`.

#### Ansible
```bash
# create virtualenv
virtualenv -p python3.8 .venv
source .venv/bin/activate

# install pip dependencies
pip install -r requirements.txt

# install ansible-galaxy dependencies
ansible-galaxy install -r requirements.yml
```

#### Bootstrap testbed
1) Adjust `scenario.yml` and specify the kubernetes worker and edge node instances.

2) Run the playbook
`ansible-playbook -i inventory.gcp.yml bootstrap-kubeedge-gcp.yml`
