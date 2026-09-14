# Recorded execution environments

The principal local execution environment used Python 3.11.14, PyTorch 2.6.0+cu118, CUDA 11.8, Transformers 4.57.6, SAE Lens 6.37.6, TransformerLens 2.17.0, Datasets 4.6.1, Accelerate 1.12.0, NumPy 1.26.4, and scikit-learn 1.8.0 on an NVIDIA RTX A5000 with 24 GiB VRAM.

Scanner versions differed across finalized experiment stages:

- the original Model 1 environment recorded Semgrep 1.154.0;
- later revision and cross-model scans used Linux/Colab environments, with final Model 2 and Model 3 protocols recording Semgrep 1.175.0.

Use the scanner requirement associated with the protocol being reproduced. Do not substitute a newer ruleset and describe the result as an exact reproduction.

The original `pip_freeze.txt` is not copied because it contains unrelated environment packages and a local editable dependency. The two curated requirement files list the dependencies needed by the released workflows.
