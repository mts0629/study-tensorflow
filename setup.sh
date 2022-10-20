#!/bin/bash

venv_name=.venv

virtualenv ${venv_name}

source ${venv_name}/bin/activate

pip install -r requrements.txt

deactivate

