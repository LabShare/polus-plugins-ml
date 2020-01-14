/**
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
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

using namespace std;

int main(int argc, char ** argv) {
	/**
	 * The errors and informational messages are outputted to the log file 
	 */
	ofstream logFile;
	logFile.open("Setting.txt");

	string filePath, outputPath;
	int K,convThreshold,DimLowSpace;
	float sampleRate;
	bool randominitializing;

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
		else if (string(argv[i])=="--convThreshold") convThreshold=atoi(argv[i+1]);
		else if (string(argv[i])=="--DimLowSpace") DimLowSpace=atoi(argv[i+1]);
		else if (string(argv[i])=="--randominitializing") randominitializing=argv[i+1];
		else if (string(argv[i])=="--outputPath"){
			boost::filesystem::path p(argv[i+1]);

			if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
			{
				logFile << "Incorrect output path";
				return 1;
			}

			boost::filesystem::path joinedPath = p / boost::filesystem::path("ProjectedData_EmbeddedSpace.csv");
			outputPath = joinedPath.string();

		}
	}

	/**
	 * The full path to the input file containig the dataset.
	 */
	//	string filePath = argv[1]; 
	logFile<<"The full path to the input file: "<< filePath<<endl;
	logFile<<"The full path to the output file: "<< outputPath<<endl;
	/**
	 * K in K-NN that means the desired number of Nearest Neighbours to be computed.
	 */
	//	const int K = atoi(argv[2]); 
	logFile<<"The desired number of NN to be computed: "<< K <<endl;
	/**
	 * The rate at which we do sampling. This parameter plays a key role in the performance.
	 * This parameter is a trades-off between the performance and the accuracy of the results.
	 * Values closer to 1 provides more accurate results but the execution takes longer.
	 */
	//	float sampleRate = stof(argv[3]);
	logFile<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl;  
	/**
	 * Convergance Threshold. A fixed integer is used here instead of delta*N*K.
	 */
	//	const int convThreshold = atoi(argv[4]); 
	logFile<<"The convergance threshold: "<< convThreshold <<endl; 
	/**
	 * Dimension of Low-D Space. 
	 */
	//	const int DimLowSpace = atoi(argv[5]);
	logFile<<"The Dimension of Low-D Space: "<< DimLowSpace <<endl; 
	/**
	 * Defining the Method for Initialization of data in low-D space
	 */	 
	//	bool randominitializing = argv[6]; 
	logFile<<"Random Initialization of Points in Low-D Space: "<< randominitializing <<endl; 	
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
	//--------------------------------------------------------------------------  
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
	 * A 2D Array containing the entire input dataset (read from filePath).
	 */
	double** data = new double*[N];
	for (int i = 0; i < N; ++i) { data[i] = new double[Dim]; }
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
	 * @return data A 2D Array containing the entire input dataset (read from filePath).	 
	 */
	computeKNNs(filePath, N, Dim, K, sampleRate, convThreshold,B_Index,B_Dist,data, logFile);

	int* B_Index_Min = new int[N];
	double* B_Dist_Min = new double[N];
	/**
	 * Compute B_Index and B_Dist for the closest point 
	 * @param B_Index indices of K-NN for each data point 	
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	 	 	 	 
	 * @return B_Index_Min B_Index for the closest point 
	 * @return B_Dist_Min B_Dist for the corresponding B_Index_Min
	 */
	findMin(B_Index,B_Dist, N,K,B_Index_Min,B_Dist_Min);

	double target=log2(K);
	double* SigmaValues = new double[N];
	/**
	 * Compute SigmaValues for each data point iteratively
	 * @param target equals to log2(K)
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param B_Dist_Min B_Dist for the corresponding B_Index_Min
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	
	 * @return SigmaValues An array of Sigma Values for data 	 	 	 
	 */
	findSigma(target,B_Dist, B_Dist_Min,SigmaValues, N, K);

	/**
	 * WFinal contains undirected weights (similarities) between the NN points in the high-D space 
	 */
	double** WFinal = new double*[N];
	for (int i = 0; i < N; ++i) { WFinal[i] = new double[K]; }	

	for (int i = 0; i < N; ++i){
		for (int j = 0; j < K; ++j){
			WFinal[i][j]=-1;
		}
	}
	/**
	 * adjacencyMatrix is simply WFinal but in the form of a matrix of size NxN 
	 */
	float** adjacencyMatrix = new float*[N];
	for (int i = 0; i < N; ++i) { adjacencyMatrix[i] = new float[N]; }
	/**
	 * diagonal matrix contains information about the degree of each vertex 
	 */
	float** degreeMatrix = new float*[N];
	for (int i = 0; i < N; ++i) { degreeMatrix[i] = new float[N]; }

	for (int i = 0; i < N; ++i){
		for (int j = 0; j < N; ++j){
			adjacencyMatrix[i][j]=0;
			degreeMatrix[i][j]=0;
		}
	}

	double WFinalMax=0;
	for (int i = 0; i < N; ++i) {
		double W, WT, tmp;
		for (int j = 0; j < K; ++j) {	
			if (WFinal[i][j]<0) {		

				W=exp((B_Dist_Min[i]-B_Dist[i][j])/SigmaValues[i]);
				int point2=B_Index[i][j]; 
				WT=exp((B_Dist_Min[point2]-B_Dist[i][j])/SigmaValues[point2]);
				tmp=W+WT-W*WT;
				WFinal[i][j]=tmp;
				if (tmp>WFinalMax) WFinalMax=tmp;

				int index=-1;  		
				for (int jj = 0; jj < K; ++jj) {		
					if (B_Index[point2][jj]==i) {index=jj;  break;}
				} 	
				if (index!=-1) WFinal[point2][index]=tmp;

				adjacencyMatrix[i][point2]=tmp;
				adjacencyMatrix[point2][i]=tmp;
			}
		}
	}

	//--------------------------------------------------------------------------   	
	logFile<<"------------Setting Low-D Space Design------------"<<endl;
	/**
	 * sizesLowSpace is an array with Min and Max values for Low-D space 
	 */
	double** sizesLowSpace = new double*[DimLowSpace];
	for (int i = 0; i < DimLowSpace; ++i) { sizesLowSpace[i] = new double[2]; }

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

	//--------------------------------------------------------------------------   
	logFile<<"------------Starting Initialization in the Low-D Space------------"<<endl;
	/**
	 * Compute SigmaValues for each data point iteratively
	 * @param randominitializing the methodology for Initialization of data in low-D space
	 * @param logFile contains the errors and informational messages 
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	
	 * @param WFinal contains undirected weights (similarities) between the NN points in the high-D space 
	 * @param degreeMatrix contains information about the degree of each vertex	 
	 * @param adjacencyMatrix contains undirected weights (similarities) in the form of a matrix of size NxN
	 * @param DimLowSpace Dimension of Low-D space 	 
	 * @param sizesLowSpace an array with Min and Max values for Low-D space 	 
	 * @return locationLowSpace is the coordinates of the points in the low-D space	 	 	 
	 */
	Initialization (randominitializing, locationLowSpace, logFile, N, K, WFinal, degreeMatrix, adjacencyMatrix, DimLowSpace, sizesLowSpace);

	//--------------------------------------------------------------------------   
	logFile<<"------------Starting Solution for Stochastic Gradient Descent (SGD)------------"<<endl;

	const float aValue=1.93;
	const float bValue=0.79;
	float alpha=1.0;  
	const int n_epochs=200;

	int** tmpIndex = new int*[N];
	for (int i = 0; i < N; ++i) { tmpIndex[i] = new int[K]; }	

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {
			tmpIndex[i][j]=1;
		}
	}    
	vector<float> head, tail;
	vector<float> epochs_per_sample;

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {	 
			if (tmpIndex[i][j]==1) {
				head.push_back(i);
				int point2=B_Index[i][j];
				tail.push_back(point2);
				epochs_per_sample.push_back(WFinalMax/WFinal[i][j]);   
				tmpIndex[i][j]=0;

				int index=-1;  		
				for (int jj = 0; jj < K; ++jj) {		
					if (B_Index[point2][jj]==i) {index=jj;  break;}
				} 	
				if (index!=-1) tmpIndex[point2][index]=0;
			}
		}
	}

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

	int move_other=1; 

	for (int n = 0; n < n_epochs; ++n) {  	    	  
		for (int i = 0; i < edgeCounts; ++i) {  	
			if (epoch_of_next_sample[i] <= n){ 	

				int headIndex = head[i];   
				int tailIndex = tail[i];  

				double dist_squared=0;
				for (int jj = 0; jj < DimLowSpace; ++jj) {  
					dist_squared += pow(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj],2);
				}
				
				double coeff;
				if (dist_squared==0) continue;  //Be Checked Later!!
				else {coeff= -2*aValue*bValue*pow(dist_squared,bValue-1)/(1+aValue*pow(dist_squared,bValue)); }
			      		
				for (int jj = 0; jj < DimLowSpace; ++jj) { 
					locationLowSpace[headIndex][jj] += alpha* clip(coeff*(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj]));

					
					if (locationLowSpace[headIndex][jj] < MinDimLowDSpace) locationLowSpace[headIndex][jj] = MinDimLowDSpace;
					else if (locationLowSpace[headIndex][jj] > MaxDimLowDSpace) locationLowSpace[headIndex][jj] = MaxDimLowDSpace;
					if (move_other==1) 	{
						locationLowSpace[tailIndex][jj] += -alpha* clip(coeff*(locationLowSpace[headIndex][jj]-locationLowSpace[tailIndex][jj]));
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

					double coeff2=2.0*bValue/((0.001+dist_squared2)*(1+aValue*pow(dist_squared2,bValue))); 

					for (int jj = 0; jj < DimLowSpace; ++jj) {  
						if  (coeff2 > 0) {
							locationLowSpace[headIndex][jj] += alpha*clip(coeff2*(locationLowSpace[headIndex][jj]-locationLowSpace[randomIndex][jj]));
							if (locationLowSpace[headIndex][jj] < MinDimLowDSpace) locationLowSpace[headIndex][jj] = MinDimLowDSpace;
							else if (locationLowSpace[headIndex][jj] > MaxDimLowDSpace) locationLowSpace[headIndex][jj] = MaxDimLowDSpace;
						}
						else  {						
							locationLowSpace[headIndex][jj] += alpha*4; 
							if (locationLowSpace[headIndex][jj] < MinDimLowDSpace) locationLowSpace[headIndex][jj] = MinDimLowDSpace;
							else if (locationLowSpace[headIndex][jj] > MaxDimLowDSpace) locationLowSpace[headIndex][jj] = MaxDimLowDSpace;
						}
					}    	        
				}       	    
				epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);  
			}   	
		}    	
		alpha=1.0-((float)n)/n_epochs;    	
	}

	//-------------------------------------------------------------------------- 
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

	return 0;
}



