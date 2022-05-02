
  

This project contains a federated machine learning application that does sarcoma classification based on histopathology slides. We use [NVIDIA NVFlare](https://github.com/NVIDIA/NVFlare) to federate the application.

  

The initial idea (and code) came from [DeepPATH](https://github.com/ncoudray/DeepPATH)

## References and Reading

 1. https://nvidia.github.io/NVFlare/installation.html
 2. https://nvidia.github.io/NVFlare/quickstart.html
 3. https://nvidia.github.io/NVFlare/examples/hello_tf2.html#train-the-model-federated

## Requirements

 1. Python 3.8 - only tested with Python 3.8. Note, that in the NVIDIA Flare documentation it currently states: "NVIDIA FLARE requires Python 3.8. It may work with Python 3.7 but currently is not compatible with Python 3.9 and above."

## Installation

 Add to your .bash_profile (this was required due to an Out of Memory error with tensorflow):

    export TF_GPU_ALLOCATOR=cuda_malloc_async
    export OPENBLAS_NUM_THREADS=1   # required for color normalization process

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

Modification of the client config includes specifying a baseimage as shown here:

    "baseimage": "/path/to/baseimage.jpg"
    
This file is used in the Vahadane color normalization process which aligns/normalizes all of the images across different color variations due to staining and scanning differences. The baseimage can be selected from any image but it is suggested to be a good representation of a complete slide or tile that may represent the "norm" across all of the user's slides. 

## Execution

Process description: The client references local digital slides, which it tiles into smaller images. These images are in turn used by the client to train a local neural network model. After a user-defined number of epochs, the client passes the model weights back to the server, which aggregates all the client model weights into a single model. This single model is then used as the basis for the next round of training till the user-defined number of training rounds is completed.

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

## Shutdown

To shutdown the servers use the following command in the admin console:

    shutdown all
    [or]
    shutdown client
    shutdown server

## Developers

A makefile is used for code formatting.

    make format # to format the files
    make lint # to test that the code passes the lint check

## Tensorboard streaming

Currently we only have support to view learning metrics directly from the client site. To enable this, the user must first configure [Tensorboard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) by setting the desired kwargs to the `tensorboard` parameter for the executor/SimpleTrainer in the client configuration.

Next, the user must start tensorboard on the node running the client training:

    tensorboard --logdir=/path/to/configured/logdir

If the training is running on a remote machine, the user will need to forward ports appropriately:

    ssh user@remote -L 6006:remote:6006

Then visit `localhost:6006` in your local machine's web browser to view training metrics.
