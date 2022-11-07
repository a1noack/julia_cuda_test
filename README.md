# Julia Cuda Test

Testing the speedup of CUDA.jl for matrix operations.

Run the following command in the Talapas terminal to start an interactive job on a single node w/ 1 GPU:

`srun --account=<pirg name> --pty --partition=gpu --mem=30G --time=240 --nodes=1 --ntasks-per-node=1 --gpus=1 bash`
