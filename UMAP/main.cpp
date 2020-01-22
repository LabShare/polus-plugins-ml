/**
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * This code is an implementation of UMAP algorithm for dimension reduction. 
 * The reference paper is “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction“, by McInnes et al., 2018 (https://arxiv.org/abs/1802.03426)
 * Jan 2020
 */

#include <vector>
#include <iostream>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>      
#include <list>
#include <string>
#include <math.h>
#include <fstream>
#include <float.h>
#include <boost/filesystem.hpp>
#include "KNN_Serial_Code.h"
#include "highDimComputes.h"
#include "Initialization.h"
#include <exception>
#include <sstream>

using namespace std;

int main(int argc, char ** argv) {
	/**
	 * The errors and informational messages are outputted to the log file 
	 */
	ofstream logFile;
	string logFileName="Setting.txt";
	logFile.open(logFileName);
	/**
	 * The input parameters are read from command line which are as follow.
	 * filePath: The full path to the input file containig the dataset.
	 * outputPath: The full path to the output csv file containing the coordinates of data in the embedding space.
	 * K: K in K-NN that means the desired number of Nearest Neighbours to be computed.
	 * sampleRate: The rate at which we do sampling. This parameter plays a key role in the performance.
	 * This parameter is a trades-off between the performance and the accuracy of the results.
	 * Values closer to 1 provides more accurate results but the execution takes longer.
	 * DimLowSpace: Dimension of Low-D or embedding space (usually 1,2,or 3).
	 * randomInitializing: Defining the Method for Initialization of data in low-D space
	 *  nEpochs: is the number of training epochs to be used in optimizing. Larger values result in more accurate embeddings
	 */
	string filePath, outputPath, LogoutputPath;
	int K,DimLowSpace,nEpochs;
	float sampleRate;
	bool randomInitializing;

	for (int i=1; i<argc;++i){
		if (string(argv[i])=="--inputPath") {
			string inputPath=argv[i+1];

			if(!boost::filesystem::exists(inputPath) || !boost::filesystem::is_directory(inputPath))
			{
				logFile << "Incorrect input path";
				return 1;
			}

			const std::string ext = ".csv";
			boost::filesystem::recursive_directory_iterator it(inputPath);
			boost::filesystem::recursive_directory_iterator endit;

			bool fileFound = false;
			while(it != endit) {
				if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ext){
					fileFound = true;
					filePath = it->path().string();
					break;
				}
				++it;
			}
			if (!fileFound){
				logFile << "CSV file is not found in the input path";
				return 1;
			}
		}
		else if (string(argv[i])=="--K") K=atoi(argv[i+1]);
		else if (string(argv[i])=="--sampleRate") sampleRate=stof(argv[i+1]);
		//else if (string(argv[i])=="--convThreshold") convThreshold=atoi(argv[i+1]);
		else if (string(argv[i])=="--DimLowSpace") DimLowSpace=atoi(argv[i+1]);
		else if (string(argv[i])=="--randomInitializing") {
			std::stringstream ss(argv[i+1]);
			ss >> std::boolalpha >> randomInitializing;
		}
		else if (string(argv[i])=="--outputPath"){
			boost::filesystem::path p(argv[i+1]);

			if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
			{
				logFile << "Incorrect output path";
				return 1;
			}

			LogoutputPath=argv[i+1];
			boost::filesystem::path joinedPath = p / boost::filesystem::path("ProjectedData_EmbeddedSpace.csv");
			outputPath = joinedPath.string();

		}
		else if (string(argv[i])=="--nEpochs") nEpochs=atoi(argv[i+1]);
	}
	logFile<<"------------The following Input Arguments were read------------"<<endl;
	logFile<<"The full path to the input file: "<< filePath<<endl;
	logFile<<"The full path to the output file: "<< outputPath<<endl;
	logFile<<"The desired number of NN to be computed: "<< K <<endl;
	logFile<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl;  
	//logFile<<"The convergance threshold: "<< convThreshold <<endl; 
	logFile<<"The Dimension of Low-D Space: "<< DimLowSpace <<endl; 
	logFile<<"Random Initialization of Points in Low-D Space: "<< randomInitializing <<endl; 
	logFile<<"The number of training epochs: "<< nEpochs <<endl; 	
	/**
	 * Size of Dataset without the header (i.e.(#Rows in dataset)-1).
	 */
	string cmd="wc -l "+filePath;
	string outputCmd = exec(cmd.c_str());
	const int N=stoi(outputCmd.substr(0, outputCmd.find(" ")))-1;
	logFile<<"The Dimension of Dataset Records (Number of Rows in inputfile w/o header ): "<< N <<endl;
	/**
	 * Dimension of Dataset (#Columns)
	 */
	int Dim;
	string cmd2="head -n 1 "+ filePath + " |tr '\\,' '\\n' |wc -l ";
	Dim = stoi(exec(cmd2.c_str())); 
	logFile<<"The Dimension of Dataset Features(Number of Columns in inputfile): "<< Dim <<endl;

	logFile<<"------------END of INPUT READING------------"<< endl;
	srand(17);			 
	/**
	 * convThreshold: Convergance Threshold of K-NN. A fixed integer is used here instead of delta*N*K. 
	 */		 
	const int convThreshold=5;
	/**
	 * indices of K-NN for each data point
	 */
	int** B_Index = new int*[N];
	for (int i = 0; i < N; ++i) { B_Index[i] = new int[K]; }	
	/**
	 * corresponding distance for K-NN indices stored in B_Index
	 */
	double** B_Dist = new double*[N];
	for (int i = 0; i < N; ++i) { B_Dist[i] = new double[K]; }

	/**
	 * Compute K-NN following the algorithm for shared-memory K-NN
	 * @param filePath The full path to the input file containig the dataset.
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1).	 
	 * @param Dim Dimension of Dataset (#Columns) 
	 * @param K the desired number of Nearest Neighbours to be computed
	 * @param sampleRate The rate at which we do sampling
	 * @param convThreshold Convergance Threshold
	 * @param logFile The errors and informational messages are outputted to the log file 	 
	 * @return B_Index indices of K-NN for each data point 	 
	 * @return B_Dist corresponding distance for K-NN indices stored in B_Index	 
	 */
	computeKNNs(filePath, N, Dim, K, sampleRate, convThreshold,B_Index,B_Dist, logFile);

	int* B_Index_Min = new int[N];
	double* B_Dist_Min = new double[N];
	/**
	 * Compute B_Index and B_Dist for the closest points (K-NNs) 
	 * @param B_Index indices of K-NN for each data point 	
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	 	 	 	 
	 * @return B_Index_Min B_Index for the closest point 
	 * @return B_Dist_Min B_Dist for the corresponding B_Index_Min
	 */
	findMin(B_Index,B_Dist, N,K,B_Index_Min,B_Dist_Min);

	double* SigmaValues = new double[N];
	/**
	 * Compute SigmaValues for each data point (Smooth approximator to K-NN distance) iteratively
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param B_Dist_Min B_Dist for the corresponding B_Index_Min
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	
	 * @return SigmaValues An array of Sigma Values for data 	 	 	 
	 */
	findSigma(B_Dist, B_Dist_Min,SigmaValues, N, K);

	/**
	 * adjacencyMatrixA is directed weight (similarity) function for for all the edges in the high-D space 
	 */
	float** adjacencyMatrixA = new float*[N];
	for (int i = 0; i < N; ++i) { adjacencyMatrixA[i] = new float[N]; }

	for (int i = 0; i < N; ++i){
		for (int j = 0; j < N; ++j){
			adjacencyMatrixA[i][j]=0;
		}
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {			
			int point2=B_Index[i][j]; 
			adjacencyMatrixA[i][point2]=exp((B_Dist_Min[i]-B_Dist[i][j])/SigmaValues[i]);			
		}
	}		

	/**
	 * adjacencyMatrixAT is the transpose of adjacencyMatrixA 
	 */
	float** adjacencyMatrixAT = new float*[N];
	for (int i = 0; i < N; ++i) { adjacencyMatrixAT[i] = new float[N]; }

	/**
	 * Initializing adjacencyMatrixAT
	 */		
	for (int i = 0; i < N; ++i){
		adjacencyMatrixAT[i][i]=adjacencyMatrixA[i][i];
	}

	for (int i = 0; i < N-1; ++i){			
		for (int j = i+1; j < N; ++j){
			adjacencyMatrixAT[j][i]= adjacencyMatrixA[i][j];
			adjacencyMatrixAT[i][j]= adjacencyMatrixA[j][i];         
		}
	}

	/**
	 * adjacencyMatrixB is undirected weights (similarities) function for all the edges in the high-D space 
	 */
	float** adjacencyMatrixB = new float*[N];
	for (int i = 0; i < N; ++i) { adjacencyMatrixB[i] = new float[N]; }	

	float MaxWeight=0;
	for (int i = 0; i < N; ++i){			
		for (int j = 0; j < N; ++j){
			float tmp = adjacencyMatrixA[i][j]+adjacencyMatrixAT[i][j]-adjacencyMatrixA[i][j]*adjacencyMatrixAT[i][j];  
			adjacencyMatrixB[i][j]=tmp;
			if (tmp > MaxWeight) MaxWeight=tmp;       
		}
	}	

	delete[] adjacencyMatrixAT, adjacencyMatrixA;  

	logFile<<"------------Setting Low-D Space Design------------"<<endl;
	/**
	 * sizesLowSpace is an array with Min (MinDimLowDSpace) and Max (MaxDimLowDSpace) values for Low-D space 
	 */
	double** sizesLowSpace = new double*[DimLowSpace];
	for (int i = 0; i < DimLowSpace; ++i) { sizesLowSpace[i] = new double[2]; }
	/**
	 * By deafult, low-D space dimensions are between -10 and 10 
	 */
	int MinDimLowDSpace=-10;
	int MaxDimLowDSpace=10;    

	for (int i = 0; i < DimLowSpace; ++i){    
		sizesLowSpace[i][0]=MinDimLowDSpace;
		sizesLowSpace[i][1]=MaxDimLowDSpace;	
	}

	/**
	 * locationLowSpace is the coordinates of the points in the low-D space  
	 */
	double** locationLowSpace = new double*[N];
	for (int i = 0; i < N; ++i) { locationLowSpace[i] = new double[DimLowSpace]; }    

	logFile<<"------------Starting Initialization in the Low-D Space------------"<<endl;
	/**
	 * Initializes the data points in low-D space
	 * @param randomInitializing the methodology for Initialization of data in low-D space
	 * @param logFile contains the errors and informational messages 
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param adjacencyMatrixB contains undirected weights (similarities) in the form of a matrix of size NxN
	 * @param DimLowSpace Dimension of Low-D space 	 
	 * @param sizesLowSpace an array with Min and Max values for Low-D space 	 
	 * @return locationLowSpace is the coordinates of the points in the low-D space	 	 	 
	 */
	Initialization (randomInitializing, locationLowSpace, logFile, N, adjacencyMatrixB, DimLowSpace, sizesLowSpace);

	logFile<<"------------Starting Solution for Stochastic Gradient Descent (SGD)------------"<<endl;
	/**
	 * Hyper-Parameters a and b. Assuming that min_dist = 0.001 
	 */
	const float aValue=1.93;
	const float bValue=0.79;
	/**
	 *  alpha is the Initial learning rate for the SGD. alpha starts from 1 and decreases in each epoch iteration
	 */	
	float alpha=1.0;  
	/**
	 * epochs_per_sample is a vector of edges with the values proportional to the values in adjacencyMatrixB 
	 * epochs_per_sample represents the epoch weight for edges where the edge with the highest similarity will get the value of 1 
	 * and all other edges will get a proportional epoch weight scaled from it. epochs_per_sample is used as a measure to include an edge in 
	 * SGD computations. The edge with the highest similarity will be used at every epoch iteration. 
	 * head is a vector containing the head index of the edge
	 * tail is a vector containing the tail index of the edge
	 */
	vector<float> head, tail;
	vector<float> epochs_per_sample;

	// zero approximation
	float epsilon=1e-6;

	// adjacencyMatrixB is a symmetric matrix, thus, we only search half of it
	for (int i = 0; i < N; ++i) {	
		for (int j = i+1; j < N; ++j) {	
			// We are only looking for non-zero elements
			if (adjacencyMatrixB[i][j] < epsilon) continue;

			epochs_per_sample.push_back(MaxWeight/adjacencyMatrixB[i][j]);
			head.push_back(i);
			tail.push_back(j);			 
		}
	}

	/**
	 * This section was adopted from SGD implementation at https://github.com/lmcinnes/umap/blob/8f2ef23ec835cc5071fe6351a0da8313d8e75706/umap/layouts.py#L136
	 * edgeCounts is total number of edges in the high-D space graph
	 * epoch_of_next_sample is an index of the epoch state of the edges. If it is less than epoch index, we will use the edge in the computation
	 * epoch_of_next_negative_sample is an index of the epoch state of the edges for sampling from non-connected surrounding points. 
	 * negative_sample_rate is the rate at which we sample from the non-connected surrounding points as compared to the connected edges. 
	 * Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
	 */	 
	int edgeCounts=epochs_per_sample.size();
	const int negative_sample_rate=5;
	int n_neg_samples;
	float epoch_of_next_sample[edgeCounts];    
	float epochs_per_negative_sample[edgeCounts]; 
	float epoch_of_next_negative_sample[edgeCounts];  

	for (int i = 0; i < edgeCounts; ++i) {
		epoch_of_next_sample[i]=epochs_per_sample[i];
		epochs_per_negative_sample[i]=epochs_per_sample[i]/negative_sample_rate;
		epoch_of_next_negative_sample[i]=epochs_per_negative_sample[i];
	}  
	/**
	 *  move_other is equal to 1 if not embedding new previously unseen points to low-D space
	 */
	int move_other=1; 
	/**
	 *  dEpsilon is zero approximation in double precision
	 */	
	double dEpsilon=1e-14;

	// The main training loop     
	for (int n = 0; n < nEpochs; ++n) {
	    //Loop over all edges of the graph  	    	  
		for (int i = 0; i < edgeCounts; ++i) {  	
			if (epoch_of_next_sample[i] <= n){ 	

				int headIndex = head[i];   
				int tailIndex = tail[i];  
				
				double dist_squared=0;
				for (int jj = 0; jj < DimLowSpace; ++jj) {  
					dist_squared += pow(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj],2);
				}

				double grad_coeff;
				if (dist_squared<dEpsilon) grad_coeff=0;  
				else {grad_coeff= -2.0*aValue*bValue*pow(dist_squared,bValue-1)/(1.0+aValue*pow(dist_squared,bValue)); }

				for (int jj = 0; jj < DimLowSpace; ++jj) { 
					locationLowSpace[headIndex][jj] += alpha* clip(grad_coeff*(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj]));

					if (locationLowSpace[headIndex][jj] < MinDimLowDSpace) locationLowSpace[headIndex][jj] = MinDimLowDSpace;
					else if (locationLowSpace[headIndex][jj] > MaxDimLowDSpace) locationLowSpace[headIndex][jj] = MaxDimLowDSpace;
					if (move_other==1) 	{
						locationLowSpace[tailIndex][jj] += -alpha* clip(grad_coeff*(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj]));
						if (locationLowSpace[tailIndex][jj] < MinDimLowDSpace) locationLowSpace[tailIndex][jj] = MinDimLowDSpace;
						else if (locationLowSpace[tailIndex][jj] > MaxDimLowDSpace) locationLowSpace[tailIndex][jj] = MaxDimLowDSpace;								
					}
				}

				epoch_of_next_sample[i] += epochs_per_sample[i];
				n_neg_samples = int((n - epoch_of_next_negative_sample[i])/ epochs_per_negative_sample[i]);     	      

				for (int ll = 0; ll < n_neg_samples; ++ll) {	    	
					int randomIndex = rand() % N;
					if (randomIndex==headIndex) continue;

					double dist_squared2=0;
					for (int jj = 0; jj < DimLowSpace; ++jj) {  
						dist_squared2 += pow(locationLowSpace[headIndex][jj]-locationLowSpace[randomIndex][jj],2);
					} 

					double grad_coeff2;
					if (dist_squared2 < dEpsilon) grad_coeff2=0; 
					else{ grad_coeff2 = 2.0*bValue/((0.001+dist_squared2)*(1.0+aValue*pow(dist_squared2,bValue))); }

					for (int jj = 0; jj < DimLowSpace; ++jj) {  
						if  (grad_coeff2 > 0) {
							locationLowSpace[headIndex][jj] += alpha*clip(grad_coeff2*(locationLowSpace[headIndex][jj]-locationLowSpace[randomIndex][jj]));
						} else  {						
							locationLowSpace[headIndex][jj] += alpha*4.0; 
						}
						if (locationLowSpace[headIndex][jj] < MinDimLowDSpace) locationLowSpace[headIndex][jj] = MinDimLowDSpace;
						else if (locationLowSpace[headIndex][jj] > MaxDimLowDSpace) locationLowSpace[headIndex][jj] = MaxDimLowDSpace;
					}    	        
				}       	    
				epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);  
			}   	
		}    	
		alpha=1.0-((float)n)/nEpochs;    	
	}

	logFile<<"------------Starting Outputing the Results------------"<<endl;
	/**
	 * Output the coordinates of the projected data in the low-D space
	 */ 
	ofstream embeddedSpacefile;
	embeddedSpacefile.open(outputPath);

	for (int j = 0; j < DimLowSpace; ++j) {
		if (j != DimLowSpace-1) embeddedSpacefile<<"Dimension"<<j+1<<",";
		else embeddedSpacefile<<"Dimension"<<j+1<<endl;
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DimLowSpace; ++j) {		
			if (j==DimLowSpace-1) {
				embeddedSpacefile<< locationLowSpace[i][j]<<endl;}
			else {embeddedSpacefile<< locationLowSpace[i][j]<<",";}
		}
	}

	embeddedSpacefile.close();
	logFile.close();
	/**
	 * copy Logfile to the file system which could be accessed outside the docker container
	 */ 
	string cmd3="cp "+logFileName+"  "+LogoutputPath+ " 2>&1 /dev/null";
	string outputCmd3 = exec(cmd3.c_str());

	return 0;
}



