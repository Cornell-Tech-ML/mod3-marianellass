# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

### 3.1 and 3.2 Parallel Analytics 

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (163)
  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        if np.array_equal(in_shape, out_shape) and np.array_equal(           | 
            in_strides, out_strides                                          | 
        ):                                                                   | 
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])                                   | 
        else:                                                                | 
            for i in prange(len(out)):---------------------------------------| #1
                out_index = np.empty(MAX_DIMS, np.int32)                     | 
                in_index = np.empty(MAX_DIMS, np.int32)                      | 
                                                                             | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                ii = index_to_position(out_index, out_strides)               | 
                                                                             | 
                j = int(index_to_position(in_index, in_strides))             | 
                out[ii] = fn(in_storage[j])                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (178)
 is hoisted out of the parallel loop labelled #1 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (179)
 is hoisted out of the parallel loop labelled #1 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (214)
  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (214) 
----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        a_storage: Storage,                                                       | 
        a_shape: Shape,                                                           | 
        a_strides: Strides,                                                       | 
        b_storage: Storage,                                                       | 
        b_shape: Shape,                                                           | 
        b_strides: Strides,                                                       | 
    ) -> None:                                                                    | 
        sameShape = np.array_equal(a_shape, b_shape) and np.array_equal(          | 
            a_shape, out_shape                                                    | 
        )                                                                         | 
        sameStrides = np.array_equal(a_strides, b_strides) and np.array_equal(    | 
            a_strides, out_strides                                                | 
        )                                                                         | 
                                                                                  | 
        if sameShape and sameStrides:                                             | 
            for i in prange(len(out)):--------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                           | 
        else:                                                                     | 
            for i in prange(len(out)):--------------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                          | 
                a_index = np.empty(MAX_DIMS, np.int32)                            | 
                b_index = np.empty(MAX_DIMS, np.int32)                            | 
                                                                                  | 
                to_index(i, out_shape, out_index)                                 | 
                broadcast_index(out_index, out_shape, a_shape, a_index)           | 
                broadcast_index(out_index, out_shape, b_shape, b_index)           | 
                                                                                  | 
                out[i] = fn(                                                      | 
                    a_storage[index_to_position(a_index, a_strides)],             | 
                    b_storage[index_to_position(b_index, b_strides)],             | 
                )                                                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (237)
 is hoisted out of the parallel loop labelled #3 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (238)
 is hoisted out of the parallel loop labelled #3 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (239)
 is hoisted out of the parallel loop labelled #3 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (274)
  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (274) 
----------------------------------------------------------------------|loop #ID
    def _reduce(                                                      | 
        out: Storage,                                                 | 
        out_shape: Shape,                                             | 
        out_strides: Strides,                                         | 
        a_storage: Storage,                                           | 
        a_shape: Shape,                                               | 
        a_strides: Strides,                                           | 
        reduce_dim: int,                                              | 
    ) -> None:                                                        | 
                                                                      | 
        reduce_size = a_shape[reduce_dim]                             | 
        #reduce_stride = a_strides[reduce_dim]                        | 
                                                                      | 
        for i in prange(len(out)):------------------------------------| #4
            out_idx = np.empty(MAX_DIMS, np.int32)                    | 
                                                                      | 
            to_index(i, out_shape, out_idx)                           | 
            out_position = index_to_position(out_idx, out_strides)    | 
            temp= out[out_position]                                   | 
                                                                      | 
            for s in range(reduce_size):                              | 
                j = 0                                                 | 
                out_idx[reduce_dim] = s                               | 
                                                                      | 
                for x, stride in zip(out_idx, a_strides):             | 
                    j += x*stride                                     | 
                temp = fn(temp, a_storage[j])                         | 
            out[out_position] = temp                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (288)
 is hoisted out of the parallel loop labelled #4 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (306)
  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/marianellasalinas/workspace/mod3-marianellass/minitorch/fast_ops.py (306) 
-------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                       | 
    out: Storage,                                                  | 
    out_shape: Shape,                                              | 
    out_strides: Strides,                                          | 
    a_storage: Storage,                                            | 
    a_shape: Shape,                                                | 
    a_strides: Strides,                                            | 
    b_storage: Storage,                                            | 
    b_shape: Shape,                                                | 
    b_strides: Strides,                                            | 
) -> None:                                                         | 
    """NUMBA tensor matrix multiply function.                      | 
                                                                   | 
    Should work for any tensor shapes that broadcast as long as    | 
                                                                   | 
    ```                                                            | 
    assert a_shape[-1] == b_shape[-2]                              | 
    ```                                                            | 
                                                                   | 
    Optimizations:                                                 | 
                                                                   | 
    * Outer loop in parallel                                       | 
    * No index buffers or function calls                           | 
    * Inner loop should have no global writes, 1 multiply.         | 
                                                                   | 
                                                                   | 
    Args:                                                          | 
    ----                                                           | 
        out (Storage): storage for `out` tensor                    | 
        out_shape (Shape): shape for `out` tensor                  | 
        out_strides (Strides): strides for `out` tensor            | 
        a_storage (Storage): storage for `a` tensor                | 
        a_shape (Shape): shape for `a` tensor                      | 
        a_strides (Strides): strides for `a` tensor                | 
        b_storage (Storage): storage for `b` tensor                | 
        b_shape (Shape): shape for `b` tensor                      | 
        b_strides (Strides): strides for `b` tensor                | 
                                                                   | 
    Returns:                                                       | 
    -------                                                        | 
        None : Fills in `out`                                      | 
                                                                   | 
    """                                                            | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0         | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0         | 
                                                                   | 
    for n in prange(out_shape[0]): #loop through batch-------------| #5
        for i in range(out_shape[1]):                              | 
            for j in range(out_shape[2]):                          | 
                                                                   | 
                row = n * a_batch_stride + i * a_strides[1]        | 
                col = n * b_batch_stride + j * b_strides[2]        | 
                info = 0.0                                         | 
                                                                   | 
                for _ in range(a_shape[-1]):                       | 
                    info += a_storage[row] * b_storage[col]        | 
                    row += a_strides[2]                            | 
                    col += b_strides[1]                            | 
                                                                   | 
                out[n * out_strides[0] +                           | 
                    i * out_strides[1] +                           | 
                    j * out_strides[2]] = info                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


### 3.5: Training loss, accuracy and timing 

#### backend: cpu, dataset: split, hidden: 100, rate: 0.05, points: 50
Epoch  0  loss  4.971922824540687 correct 28
Epoch  10  loss  6.0564156952979005 correct 42
Epoch  20  loss  6.2151452932622755 correct 46
Epoch  30  loss  3.8922995653155117 correct 45
Epoch  40  loss  2.853426070026546 correct 48
Epoch  50  loss  2.787386525948941 correct 49
Epoch  60  loss  2.8503217100729565 correct 49
Epoch  70  loss  2.1806177642382134 correct 48
Epoch  80  loss  1.5277872492746654 correct 48
Epoch  90  loss  1.1118199218059854 correct 48
Epoch  100  loss  1.6343984195341652 correct 49
Epoch  110  loss  0.8694639826070102 correct 49
Epoch  120  loss  1.423403185639915 correct 49
Epoch  130  loss  0.3966510379860659 correct 49
Epoch  140  loss  2.289824356038101 correct 47
Epoch  150  loss  0.8619812425685061 correct 49
Epoch  160  loss  1.7441206190007377 correct 48
Epoch  170  loss  0.10192051227350583 correct 49
Epoch  180  loss  0.20293624728870663 correct 50
Epoch  190  loss  2.184859207875409 correct 41
Epoch  200  loss  1.0837773396826116 correct 49
Epoch  210  loss  0.07522108744448314 correct 50
Epoch  220  loss  1.0957680997923218 correct 47
Epoch  230  loss  2.744788786015795 correct 46
Epoch  240  loss  1.9980614369817415 correct 44
Epoch  250  loss  0.10904054742112854 correct 47
Epoch  260  loss  0.07411200342729603 correct 48
Epoch  270  loss  0.18399808269166162 correct 49
Epoch  280  loss  10.735761309743857 correct 48
Epoch  290  loss  1.0577077983266059 correct 48
Epoch  300  loss  6.5349832641800045 correct 36
Epoch  310  loss  2.385245209408473 correct 46
Epoch  320  loss  7.1750004297789305 correct 50
Epoch  330  loss  0.03039497198791717 correct 49
Epoch  340  loss  0.0006058621979469252 correct 50
Epoch  350  loss  0.8725645642956861 correct 49
Epoch  360  loss  0.0763341984295766 correct 46
Epoch  370  loss  13.270051408309765 correct 42
Epoch  380  loss  0.014527930138848209 correct 49
Epoch  390  loss  0.014925620580594526 correct 49
Epoch  400  loss  0.01895042071009921 correct 50
Epoch  410  loss  0.020942705557715085 correct 47
Epoch  420  loss  0.0032875822639205792 correct 49
Epoch  430  loss  0.1655377367102712 correct 50
Epoch  440  loss  1.4672243091437345 correct 47
Epoch  450  loss  0.0008488555840592742 correct 45
Epoch  460  loss  0.02176308153008666 correct 49
Epoch  470  loss  0.26799589932555834 correct 51
Epoch  480  loss  0.0008243049223472969 correct 45
Epoch  490  loss  0.0022972039051135946 correct 49
