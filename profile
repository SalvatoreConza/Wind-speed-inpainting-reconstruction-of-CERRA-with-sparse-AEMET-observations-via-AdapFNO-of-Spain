--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
         90968498 function calls (90919836 primitive calls) in 254.738 seconds

   Ordered by: internal time
   List reduced from 7764 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       86  191.191    2.223  191.191    2.223 {method 'poll' of 'select.poll' objects}
      330   20.459    0.062   24.703    0.075 /opt/conda/envs/neural_operator/lib/python3.10/site-packages/cfgrib/messages.py:442(header_values)
     3344   12.338    0.004   12.338    0.004 {method 'item' of 'torch._C.TensorBase' objects}
       22    9.247    0.420   11.003    0.500 /opt/conda/envs/neural_operator/lib/python3.10/site-packages/cfgrib/messages.py:464(subindex)
 63375547    4.193    0.000    4.193    0.000 {method 'setdefault' of 'dict' objects}
        2    2.953    1.476    2.953    1.476 {built-in method _pickle.load}
       20    2.491    0.125   16.936    0.847 /opt/conda/envs/neural_operator/lib/python3.10/site-packages/cfgrib/dataset.py:488(build_variable_components)
       30    1.843    0.061    1.843    0.061 {method 'run_backward' of 'torch._C._EngineBase' objects}
14747365/14746975    1.640    0.000    1.650    0.000 {built-in method builtins.isinstance}
        1    0.943    0.943  254.741  254.741 cli/train_global.py:1(<module>)
       16    0.749    0.047    0.750    0.047 {built-in method posix.fork}
  2520582    0.708    0.000    0.708    0.000 {method 'index' of 'list' objects}
      131    0.417    0.003    0.417    0.003 {method 'to' of 'torch._C.TensorBase' objects}
      532    0.414    0.001    0.414    0.001 {built-in method posix.read}
  5572079    0.332    0.000    0.332    0.000 {method 'append' of 'list' objects}


--------------------------------------------------------------------------------
  autograd profiler output (CUDA mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

        Because the autograd profiler uses the CUDA event API,        the CUDA time column reports approximately max(cuda_time, cpu_time).
        Please ignore this output if your code does not use CUDA.

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        10.80%       20.159s        10.80%       20.159s       20.159s       20.159s        10.80%       20.159s       20.159s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        10.75%       20.060s        10.75%       20.067s       20.067s       20.067s        10.75%       20.067s       20.067s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        10.70%       19.963s        10.70%       19.963s       19.963s       19.963s        10.70%       19.963s       19.963s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        10.67%       19.912s        10.68%       19.919s       19.919s       19.919s        10.67%       19.919s       19.919s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.68%       18.061s         9.68%       18.061s       18.061s       18.061s         9.68%       18.061s       18.061s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.55%       17.814s         9.55%       17.822s       17.822s       17.821s         9.55%       17.821s       17.821s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.39%       17.523s         9.40%       17.532s       17.532s       17.531s         9.39%       17.532s       17.532s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.36%       17.470s         9.37%       17.476s       17.476s       17.476s         9.36%       17.476s       17.476s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.35%       17.454s         9.35%       17.454s       17.454s       17.453s         9.35%       17.453s       17.453s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         9.35%       17.442s         9.35%       17.449s       17.449s       17.449s         9.35%       17.449s       17.449s             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.11%     210.225ms         0.11%     210.298ms     210.298ms     210.244ms         0.11%     210.306ms     210.306ms             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.08%     143.541ms         0.08%     143.686ms     143.686ms     143.557ms         0.08%     143.691ms     143.691ms             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.07%     137.004ms         0.07%     137.004ms     137.004ms     137.091ms         0.07%     137.091ms     137.091ms             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.07%     124.187ms         0.07%     124.187ms     124.187ms     124.260ms         0.07%     124.260ms     124.260ms             1  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.06%     118.122ms         0.06%     118.161ms     118.161ms     118.129ms         0.06%     118.165ms     118.165ms             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 186.591s
Self CUDA time total: 186.632s