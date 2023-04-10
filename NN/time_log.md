# Time logs

This file contains basic information about performance of the models and their training.

The simplified notation reads as follows:

* 2-N(2)-n(mr)-1sig

This means that the input vector is 2 rows, followed by a ``normBatch`` layer size of the input, followed by n layers of m neurons each, 'r' denoting the ``relu`` activation function. At the end there is 1 output neuron with ``sigmoid`` activation function.

***

## Model 2-2N-5(256r)-1sig

31.697556 seconds (49.66 M allocations: 3.385 GiB, 1.54% gc time, 3.39% compilation time) - 70.0%\
30.427596 seconds (46.56 M allocations: 3.222 GiB, 1.57% gc time) - 71.6%\
30.932998 seconds (46.59 M allocations: 3.223 GiB, 1.50% gc time) - 71.6%\
31.204084 seconds (46.55 M allocations: 3.222 GiB, 1.44% gc time) - 73.4%\
31.655717 seconds (46.60 M allocations: 3.223 GiB, 1.46% gc time) - 71.8%

### 1000 n per 'batch'

52.130530 seconds (47.00 M allocations: 3.229 GiB, 1.07% gc time) - 85.5%\
52.198616 seconds (47.04 M allocations: 3.230 GiB, 1.05% gc time) - 84.6%\
53.507115 seconds (47.02 M allocations: 3.229 GiB, 1.01% gc time) - 86.5%\
51.696379 seconds (47.02 M allocations: 3.229 GiB, 1.06% gc time) - 85.3%\
54.409082 seconds (47.02 M allocations: 3.229 GiB, 1.02% gc time) - 85.9%

***

## Model 2-2N-10(128r)-1sig

26.742449 seconds (78.58 M allocations: 5.546 GiB, 2.81% gc time, 3.94% compilation time) - 70.39%\
25.871774 seconds (76.17 M allocations: 5.413 GiB, 2.86% gc time) - 72.6%\
25.875591 seconds (76.12 M allocations: 5.411 GiB, 2.80% gc time) - 69.0%\
25.535802 seconds (76.14 M allocations: 5.412 GiB, 2.81% gc time) - 63.4%\
25.692058 seconds (76.14 M allocations: 5.413 GiB, 2.85% gc time) - 66.6%

***

## Model 2-2N-4(512r)-1sig

### 500 n per 'batch'

56.901125 seconds (44.49 M allocations: 2.999 GiB, 0.81% gc time, 1.46% compilation time) - 71.0%\
55.335128 seconds (42.01 M allocations: 2.863 GiB, 0.76% gc time) - 77.2%\
58.040763 seconds (42.03 M allocations: 2.864 GiB, 0.76% gc time) - 77.4%\
56.710711 seconds (42.00 M allocations: 2.863 GiB, 0.74% gc time) - 77.8%\
57.162129 seconds (42.00 M allocations: 2.863 GiB, 0.75% gc time) - 79.2%

### 1000 ns per 'batch'

83.926185 seconds (42.39 M allocations: 2.869 GiB, 0.64% gc time) - 89.0%\
83.870246 seconds (42.39 M allocations: 2.869 GiB, 0.61% gc time) - 90.3%\
83.790833 seconds (42.40 M allocations: 2.869 GiB, 0.61% gc time) - 87.8%\
86.590092 seconds (42.40 M allocations: 2.869 GiB, 0.60% gc time) - 89.6%\
85.640043 seconds (42.38 M allocations: 2.869 GiB, 0.60% gc time) - 88.4%
