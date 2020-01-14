===================================
UMAP Code for Shared-Memory Systems
===================================

Please consider the following instruction for the execution of UMAP Code 
for Shared-Memory systems. Please refer to `this link <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/745537586/UMAP+Implementations+in+C+>`_ for detailed theoretical background about PCA.

-------------------------------
Installing the Required Library
-------------------------------

UMAP requires two external libraries of Boost and Armadillo for the execution. The steps for installing Boost library are explained below.
 
.. code:: bash
    
    wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
    tar xfz boost_1_71_0.tar.gz 
    cd boost_1_71_0/
    ./bootstrap.sh
    ./b2
    export LD_LIBRARY_PATH=currentpath/stage/lib:$LD_LIBRARY_PATH

It is recommended to include the last line in the above into .bashrc file at home directory. 

The Armadillo library can be also installed using the following command.

.. code:: bash
    sudo apt-get -y install libarmadillo-dev

-----------------
Runtime Arguments
-----------------

The code required the following parameters as the input.

1- ``filePath``: The full path to the input csv file containig the dataset.
2- ``K``: the desired number of Nearest Neighbours to be computed.
3- ``sampleRate``: the rate at which we do sampling in K-NN algorithm. This parameter plays a key role
   in the performance. This parameter is a trades-off between the performance 
   and the accuracy of the results. Values closer to 1 provides more accurate
   results but the execution instead takes longer.    
4- ``convThreshold``: An integer that controls the convergence of the model in K-NN algorithm. A fixed
   integer is used here instead of delta*N*K that was given in the paper.  
5- ``DimLowSpace``: Dimension of Low-D Space (Usually is between 1 to 3)
6- ``randominitializing``: If set to true, the positions of data in the lower dimension space are initialized randomly; and if set to false, the positions are defined by solving Laplacian matrix using Armadillo library.  

-----------
The Outputs
-----------

The code produces the following output files:

1- ``ProjectedData_EmbeddedSpace.csv``: The coordinates of the projected input data in the lower dimension space.
2- ``Setting.txt``: The logging file containing the error and informational messages. 

------------------------------
An Example of Running the code
------------------------------

.. code:: bash

    ulimit -s unlimited
    g++ -I/path to boost directory/boost_1_71_0  main.cpp KNN_Serial_Code.cpp highDimComputes.cpp Initialization.cpp -o a.out -O2 -larmadillo -L/path to boost directory/boost_1_71_0/stage/lib -lboost_iostreams -lboost_system -lboost_filesystem
    time ./a.out --inputPath . --K 9 --sampleRate 0.8 --convThreshold 5 --DimLowSpace 2 --randominitializing true --outputPath .

