pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y
pip install -c /workspace/main/lm-engine/scripts/prime-intellect/constraints.txt nvidia-cutlass-dsl

pip install quack-kernels==0.3.9
