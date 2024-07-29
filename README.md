# CudaMatrixMath
Small implementation of using cuda to preform matrix multiplication

Toyed around with different implementations of how to speed up matrix multiplication via the gpu

Orignally started with simple approach where each thread corresponding to each index in the result matrix.

The first speedup I noticed was when I decided to keep the sum stored in the local cache to avoid expensive memory calls

Next to improve the speed, I toyed around the idea of reading larger chunks using int4s. This reduced the number of memory calls by a factor of 4, through the technique of vectorised memory access

Lastly I settled on the final solution of just using a shared memory tile technique, where I made each thread access portions of the matrix into a shared center. This allowed for the program to remove all redudant memory calls within block.

For future updates, I was looking to combine the shared memory cache with the vectorised memory access.
Where I designate each thread to pull a unique section of int4 data. 
This would help reduce the number of memory calls even greater, since the shared memory tiles could be made larger with the GPU I am currently using

Final Results
Simple Memory Elapsed time: 37642.3 ms
Shared Memory Elapsed time: 12722.1 ms
