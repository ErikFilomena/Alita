# Introducing Alita

If you have ever taken some machine learning class in which neural networks were involved you probably never had to actually build one. There are many good quality
ones out there that are faster and easier to use than anything one can possibly code up by himself. And yet, ever since I first saw a neural network in action I 
wanted to try and code one for myself. 

The first time I coded a neural net from scratch I did in Octave. This is an easy way to do it because the matrix operations, the backbone of a neural network, are
easy to do. This is nive to learn the fundamental algorithim, but will not scale well for bigger problems (unless one uses an actual NN package). The next one I coded
was in C++ with [OpenBLas](https://www.openblas.net/) for the linear algebra. As one codes further and understands the algorithim better one can make many improvements.
This neural network was pretty neat in my opinion, was easy to customize and I was overal satisfied. But it was slow, and the main reason for that is because I was using the 
CPU for my computations.

The next itaration of what I call Alita (a tribute to the series Battle Angel Alita) was done for a project in a deep learning class I took at FSU. That was a great oportunity
to learn more about machine learning in general, but in special, about the inner working of neural networks. By this point Alita was using [CUDA](https://developer.nvidia.com/cuda-zone)
to do the heavy lifting. It was much faster than before and it achieved pretty good results when we tested it. But it was still slower than anything that one can easily use
on Python with a few lines of codes. Additionally, due to time cosntraints, many things were done in a way that further optimization was really hard.

This brings us to the current iteration. With the experience I gained in the past year or so I am coding my own neural network library in a way that I believe can be really
powerful, scallable, and can be used in my situations. As a by product I will probably be developing things that are useful for other applications as well.

The codes in this repository will eventually constitute Alita's core, that is the basic "under the hood" stuff that is necessary to run a neural network, and potentially other 
ML applications.
