/**
 * Initializing the data points in the low-D space
 */

#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;


void Initialization (bool randominitializing, double** locationLowSpace, ofstream& logFile, int N, int K, double** WFinal, float** degreeMatrix, float** adjacencyMatrix, int DimLowSpace, double** sizesLowSpace){

	srand(17);

	if (!randominitializing){
		try{
			logFile<<" Spectral Initialization of Data in Lower Space"<<endl;
			for (int i = 0; i < N; ++i) {
				float sum=0;
				for (int j = 0; j < K; ++j) {	
					sum+=WFinal[i][j];
				}
				degreeMatrix[i][i]=1.0/sqrt(sum);
			}

			float* aux_mem = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem[j*N+i]=degreeMatrix[i][j];       //column-wise
				}
			}
			delete [] degreeMatrix;

			fmat matDegreeMatrix(aux_mem,N,N,false,true);
			sp_fmat spmatDegreeMatrix(matDegreeMatrix);    

			float* aux_mem2 = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem2[j*N+i]=adjacencyMatrix[i][j];   //column-wise
				}
			}
			delete [] adjacencyMatrix;

			fmat matadjacencyMatrix(aux_mem2,N,N,false,true);
			sp_fmat spmatadjacencyMatrix(matadjacencyMatrix);    
			sp_fmat Unity = speye<sp_fmat>(N,N); 

			sp_fmat laplacianMatrix;
			laplacianMatrix= Unity-spmatDegreeMatrix*spmatadjacencyMatrix*spmatDegreeMatrix;

			//laplacianMatrix is a Symmetric Matrix
			fvec eigval;
			fmat eigvec;
			eigs_sym(eigval, eigvec, laplacianMatrix, DimLowSpace+1 , "sm"); 

			typedef std::vector<float> stdvec;
			std::vector< std::vector<float> > tmpvector;

			for (int i = 1; i < DimLowSpace+1; ++i) {
				stdvec vectest = arma::conv_to< stdvec >::from(eigvec.col(i));
				//will throw "error: Mat::col(): index out of bounds" if no eigvec was available
				tmpvector.push_back(vectest);  
			}

			for (int j = 0; j < DimLowSpace; ++j) {    
				for (int i = 0; i < N; ++i) {
					locationLowSpace[i][j]=tmpvector[j][i];  
					if (locationLowSpace[i][j] < sizesLowSpace[j][0]) locationLowSpace[i][j] = sizesLowSpace[j][0];
					else if (locationLowSpace[i][j] > sizesLowSpace[j][1]) locationLowSpace[i][j] = sizesLowSpace[j][1];					
				}
			}
		} catch(std::exception& e){
			logFile<<" Spectral Initialization Failed. Will proceed with random initialization."<<endl; 
			randominitializing=true ; }
	}



	if (randominitializing){
		logFile<<" Random Initialization of Data in Lower Space"<<endl;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < DimLowSpace; ++j) {
				double tmp=(double)rand()/RAND_MAX;
				locationLowSpace[i][j]=sizesLowSpace[j][0]+(sizesLowSpace[j][1]-sizesLowSpace[j][0])*tmp;
				if (locationLowSpace[i][j] < sizesLowSpace[j][0]) locationLowSpace[i][j] = sizesLowSpace[j][0];
				else if (locationLowSpace[i][j] > sizesLowSpace[j][1]) locationLowSpace[i][j] = sizesLowSpace[j][1];				
			}
		}
	}


	return;
}
