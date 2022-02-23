
# sarcoma-histo-fed

  

This project contains a federated machine learning application that does sarcoma classification based on histopathology slides. We use [NVIDIA NVFlare](https://github.com/NVIDIA/NVFlare) to federate the application.

  

The initial idea (and code) came from [DeepPATH](https://github.com/ncoudray/DeepPATH)

## References and Reading

 1. https://nvidia.github.io/NVFlare/installation.html
 2. https://nvidia.github.io/NVFlare/quickstart.html
 3. https://nvidia.github.io/NVFlare/examples/hello_tf2.html#train-the-model-federated

## Requirements

 1. Python 3.8 (software must use this version of python)

## Installation

 Add to your .bash_profile:

    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64
    export CUDA_VISIBLE_DEVICES=0
    export TF_GPU_ALLOCATOR=cuda_malloc_async

Execute (to create directory, pull code, and create a virtual environment):  

    mkdir fed-sarcoma
    cd fed-sarcoma
    git clone https://github.com/mlathara/sarcoma-histo-fed.git
    python3.8 -m venv venv
    source venv/bin/activate

Execute (to install libraries and to install NVIDIA NVFlare):
  

    cd sarcoma-histo-fed
    pip install -r requirements.txt // OR for development: pip install -r requirements-dev.txt
    cd ..
    git clone https://github.com/NVIDIA/NVFlare.git

Establish one server and one client and link configuration files

    poc -n 1
    ln -s ~/fed-sarcoma/sarcoma-histo-fed poc/admin/transfer

Modify the config/json files located in sarcoma-histo-fed/sarcoma-histo-fed/config

## Execution

Start the server, the client, and the admin in 3 separate terminals (ensuring you have the venv activated via source venv/bin/activate):

    ./poc/server/startup/start.sh
    ./poc/site-1/startup/start.sh
    ./poc/admin/startup/fl_admin.sh localhost
  *Note that user and password for admin server is 'admin'/'admin'

Execute process via the admin terminal console:  

    upload_app sarcoma-histo-fed
    set_run_number 1
    deploy_app sarcoma-histo-fed all
    start_app all







