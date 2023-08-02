This is a modified fork of huggingface/transformers with a new model added:
* ViTMAE3D

To install locally in a new virtual environment, please perform the following:
1. Clone this repository
2. Create a directory for the NMSS project files and `cd` into it
3. Create a virtual environment for the NMSS project and activate it: `python3 -m venv .venv`, `source .venv/bin/activate`
4. Change directory to the cloned transformers repo and install locally in “editable” mode: `pip install -e .`
6. Install `torch` and `monai`
7. Change directory to where your NMSS project files are, do `python3` to start an interactive python shell. `from transformers import ViTMAE3DConfig`. If there is no error, then success and you can start developing!