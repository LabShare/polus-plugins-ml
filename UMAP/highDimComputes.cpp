/**
 * Extra Functions needed for computations 
 */

#include <math.h> 
#include <float.h>
#include <iostream>
#include "highDimComputes.h"
using namespace std;

void findMin(int** B_Index,double** B_Dist, int N,int K,int* B_Index_Min,double* B_Dist_Min){

	for (int i=0;i<N;++i){
		double minValue=B_Dist[i][0];
		int minID=B_Index[i][0];

		for (int j=1; j<K; ++j){

			if (B_Dist[i][j]<minValue){
				minValue=B_Dist[i][j];
				minID=B_Index[i][j];
			}
		}
		B_Index_Min[i]=minID;
		B_Dist_Min[i]=minValue;
	}
}


void findSigma(double target, double ** B_Dist, double * B_Dist_Min, double * SigmaValues, int N, int K){
	const int iterations=10000;
	const double Error=1e-5;

	for (int i=0; i<N; ++i){
		double sigma=1;
		double low=0;
		double high=DBL_MAX;

		for (int iter=0; iter<iterations; ++iter){
			double sum=0;
			for (int j=0; j<K; ++j){
				sum += exp((B_Dist_Min[i]-B_Dist[i][j])/sigma);
			}

			if ( abs(sum-target) < Error) break;

			if (sum > target){
				high=sigma;
				sigma=(low+high)*0.5;
			}
			else{
				low=sigma;
				if (high == DBL_MAX) {
					sigma *=2;
				}
				else{
					sigma=(low+high)*0.5;
				}
			}
		}
		SigmaValues[i] = sigma;
	}
}

double clip(double value){

	const double clipLowVal=-4;
	const double clipHighVal=4;
	double returnValue;

	if (value < clipLowVal) {returnValue=clipLowVal;}
	else if ( value > clipHighVal) {returnValue=clipHighVal;}
	else {returnValue=value;}

	return returnValue;
}





