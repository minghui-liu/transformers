This is modified fork of huggingface/transformers with a new added model:
* ViTMAE3D

To install locally in a new virtual environment, please perform the following:
1. Clone this repository
2. `cd transformers`
3. `python3 -m venv .venv`
4. `source .venv/bin/activate`
5. `pip install -e .`
6. Change directory to where your NMSS project files are, do `python3` to start an interactive python shell. `from transformers import ViTMAE3DConfig`. If there is no error, then success and you can start developing!