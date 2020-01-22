/**
 * Initializing the data points in the low-D space
 */

#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;


void Initialization (bool randominitializing, double** locationLowSpace, ofstream& logFile, int N, float** adjacencyMatrix, int DimLowSpace, double** sizesLowSpace){

	srand(17);

	if (!randominitializing){
		try{
			logFile<<" Spectral Initialization of Data in Lower Space"<<endl;

			/**
			 * DegreeMatrix is a diagonal matrix contains information about the degree of each vertex 
			 * sqrtDegreeMatrix transforms the diagonal values of DegreeMatrix by 1.0/sqrt()
			 */
			float** sqrtDegreeMatrix = new float*[N];
			for (int i = 0; i < N; ++i) { sqrtDegreeMatrix[i] = new float[N]; }

			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					sqrtDegreeMatrix[i][j]=0;
				}
			}

			for (int i = 0; i < N; ++i) {
				float sum=0;
				for (int j = 0; j < N; ++j) {	
					sum+=adjacencyMatrix[i][j];
				}
				sqrtDegreeMatrix[i][i]=1.0/sqrt(sum);
			}
			/**
			 * aux_mem is the column-wise transformation of sqrtDegreeMatrix as needed by armadillo function fmat
			 */
			float* aux_mem = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem[j*N+i]=sqrtDegreeMatrix[i][j];      
				}
			}
			delete [] sqrtDegreeMatrix;
			/**
			 * Making an armadillo sparse matrix spmatDegreeMatrix from sqrtDegreeMatrix
			 */
			fmat matDegreeMatrix(aux_mem,N,N,false,true);
			sp_fmat spmatDegreeMatrix(matDegreeMatrix);    
			/**
			 * aux_mem2 is the column-wise transformation of adjacencyMatrix as needed by armadillo function fmat
			 */
			float* aux_mem2 = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem2[j*N+i]=adjacencyMatrix[i][j];   //column-wise
				}
			}
			/**
			 * Making an armadillo sparse matrix spmatadjacencyMatrix from adjacencyMatrix
			 */
			fmat matadjacencyMatrix(aux_mem2,N,N,false,true);
			sp_fmat spmatadjacencyMatrix(matadjacencyMatrix);
			/**
			 * Making an armadillo sparse matrix of identity 
			 */			    
			sp_fmat Unity = speye<sp_fmat>(N,N); 
			/**
			 * Making an armadillo sparse matrix of Laplacian 
			 */	
			sp_fmat laplacianMatrix;
			laplacianMatrix= Unity-spmatDegreeMatrix*spmatadjacencyMatrix*spmatDegreeMatrix;
			/**
			 * Solving eigenvalue and eigenvector for Laplacian matrix
			 */
			fvec eigval;
			fmat eigvec;
			eigs_sym(eigval, eigvec, laplacianMatrix, DimLowSpace+1 , "sm"); 
			/**
			 * Converting eigenvectors to tmpvector 
			 * will throw "error: Mat::col(): index out of bounds" if no eigvec was available
			 */
			typedef std::vector<float> stdvec;
			std::vector< std::vector<float> > tmpvector;

			for (int i = 1; i < DimLowSpace+1; ++i) {
				stdvec vectest = arma::conv_to< stdvec >::from(eigvec.col(i));			
				tmpvector.push_back(vectest);  
			}

			//cout<<tmpvector.size()<<"  "<<tmpvector[0].size()<<endl; 
			/**
			 * using tmpvector to intialize the locations of the points in low-D space
			 * locationLowSpace should not be outside the chosen dimensions for low-D space
			 */
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
	/**
	 * If the above procedure fails or randominitializing=1 as an input argument, the
	 * location of the points are determined randomly 
	 */
	if (randominitializing){
		logFile<<" Random Initialization of Data in low-D Space"<<endl;
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