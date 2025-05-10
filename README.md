# ML parallel practice

## 📝 Notebooks

#### TPU Examples

* (Published: 2025-05-10) [TPU_2B_GPT_JAX_Parallelism.ipynb](https://github.com/JLAI162/ML_parallel_practice/blob/main/TPU_2B_GPT_JAX_Parallelism.ipynb) – A JAX example demonstrating advanced model training on TPU v3-8 with a 2-billion parameter GPT-like model, integrating fp16 precision, gradient accumulation, activation recomputation, and both tensor and sequence parallelism. It leverages JAX's sharding and partitioning primitives for efficient distributed execution across TPU cores, optimizing memory and compute efficiency for large-scale language modeling.

* (Published: 2025-04-06) [Simple_NN_jax_TPU_DP_MP.ipynb](https://github.com/JLAI162/ML_parallel_practice/blob/main/Simple_NN_jax_TPU_DP_MP.ipynb) - A JAX example showcasing combined Data and Model Parallelism on TPU. It uses JAX's sharding features to parallelize both data and model parameters for TPU execution.

* (Published: 2025-04-06) [Simple_NN_jax_TPU_MP.ipynb](https://github.com/JLAI162/ML_parallel_practice/blob/main/Simple_NN_jax_TPU_MP.ipynb) - A simple neural network example implementing Model Parallelism on TPU using JAX. It utilizes JAX's sharding and NamedSharding features to partition and parallelize model's parameter for execution on TPU. 

* (Published: 2025-04-05) [Simple_NN_jax_TPU_DP.ipynb](https://github.com/JLAI162/ML_parallel_practice/blob/main/Simple_NN_jax_TPU_DP.ipynb) - A simple neural network example implementing Data Parallelism on TPU using JAX. It utilizes JAX's sharding and NamedSharding features to partition and parallelize data for execution on TPU. (Updated: 2025-04-06)
