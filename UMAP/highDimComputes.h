#include <string>
using namespace std;

void findMin(int** B_Index,double** B_Dist, int N,int K,int* B_Index_Min,double* B_Dist_Min);


void findSigma(double target, double ** B_Dist, double * B_Dist_Min, double * SigmaValues, int N, int K);


double clip(double value);
