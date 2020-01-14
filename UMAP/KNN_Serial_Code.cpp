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
#include <boost/iostreams/stream.hpp> 
#include "KNN_Serial_Code.h"
using namespace std;



/**
 * Read the output of linux command execution 
 * @param  cmd  is the linux command to be executed
 * @return the output from the execution of the linux command
 */
std::string exec(const char* cmd) {
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}


void computeKNNs(string filePath, const int N, const int Dim, const int K, float sampleRate, const int convThreshold,int** B_Index,double** B_Dist,double** dataPoints, ofstream& logFile){
	logFile<<"------------Starting K-NN Solution------------"<<endl;
	/**
	 * corresponding flag for K-NN indices stored in B_Index
	 */
	int** B_IsNew = new int*[N];
	for (int i = 0; i < N; ++i) { B_IsNew[i] = new int[K]; }
	/**
	 * Data structure for new[v]
	 */
	vector<int> *New_Index = new std::vector<int>[N];
	/**
	 * Data structure for REVERSE(new[v]) or new'
	 */
	vector<int> *Reverse_New_Index = new vector<int>[N];
	/**
	 * Data Structure for SAMPLE(new'[v],pk)
	 */
	vector<int> *Sampled_Reverse_New_Index = new vector<int>[N];
	/**
	 * Data Structure for new[v] U SAMPLE(new'[v],pk)
	 */
	list<int> *New_Final_List = new list<int>[N];
	/**
	 * Iterators to access data stored in the list
	 */
	list<int>::iterator it, it2, it_temp;
	/**
	 * An approximation of zero in computing distances. Two points with the distance
	 * smaller than epsilon are considered as one point.
	 */
	double epsilon = 1e-14; 
	short* allEntriesFilled = new short[N];
	/**
	 * At first, let's Read Dataset from Input File
	 */
	ifstream infile;
	infile.open(filePath);
	if (infile.fail())
	{
		logFile << "error in Opening Input File" << endl;
		return ;
	}
	/**
	 * Remove the header info
	 */
	string dummyLine;
	getline(infile, dummyLine);
	/**
	 * Reading the Entire Dataset
	 */
	//	if (argc==5){ //??
	for (int i = 0; i < N; ++i) {
		string temp, temp2;
		getline(infile, temp);
		for (int j = 0; j < Dim; ++j) {
			temp2 = temp.substr(0, temp.find(","));
			dataPoints[i][j] = atof(temp2.c_str());
			temp.erase(0, temp.find(",") + 1);
		}
	}
	//	} 
	// 	else {

	/*		for (int i = 0; i < N; ++i) {
			string temp, temp2;
			getline(infile, temp);
			for (int j = 0; j < Dim; ++j) {
			temp2 = temp.substr(0, temp.find(","));
			if (j >= colIndex1-1 && j < colIndex2) dataPoints[i][j] = atof(temp2.c_str());
			temp.erase(0, temp.find(",") + 1);
			}
			*/	//	}	
	//	}
	infile.close();
	//	if (argc == 7) Dim=colIndex2-colIndex1+1;
	/**
	 * define a seed for random generator. Using a constant value produces
	 * the same set of random numbers and is good for debugging. Alternatively,
	 * we can select the seed number randomly as srand(time(NULL))
	 */
	srand(17);
	/**
	 * Initialization of Arrays B_IsNew and B_Dist
	 */
	for (int i = 0; i < N; ++i) {
		allEntriesFilled[i]=0;
		for (int j = 0; j < K; ++j) {
			B_IsNew[i][j] = 1;
			B_Dist[i][j] = -1.0;
		}
	}
	/**
	 * Random Initialization of B_Index
	 */
	int randomIndex, iter;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {
			iter = 1;
			while (iter) {
				randomIndex = rand() % N;
				if (randomIndex != i) {
					B_Index[i][j] = randomIndex;
					iter = 0;
				}
			}
		}
	}
	/**
	 * Update list of K-NN indices for u1 (B_Index) if u2 is closer
	 * <p>
	 * This method correspondd to UPDATENN(B[u1],<u2,l,true>) in the paper
	 * </p>
	 * @param  Dist  represents B_Dist
	 * @param  Index represents B_Index
	 * @param  IsNew represents B_IsNew
	 * @param  u1    the indice of point that we want to potentially update its K-NN with the point u2
	 * @param  u2    the indice of potential K-NN fpr point u1
	 * @param  distance the spatial distance between u1 and u2
	 * @param  flag updates B_IsNew
	 * @return 1 if B_Index[u1][.] is updated, 0 otherwise
	 */
	auto UpdateNN = [&](int u1, int u2, double distance, int flag = 1) {

		if(allEntriesFilled[u1]==0){		
			for (int j = 0; j < K; j++) {	
				if (B_Dist[u1][j] < 0) {
					B_Dist[u1][j] = distance;
					B_Index[u1][j] = u2;
					B_IsNew[u1][j] = flag;
					if (j==K-1) allEntriesFilled[u1]=1;
					return 1;}
			}
		}

		else{
			for (int j = 0; j < K; j++) {
				if (B_Index[u1][j] == u2) return 0;
			}

			double max = DBL_MIN;
			int index = -1;
			for (int j = 0; j < K; j++) {
				if (B_Dist[u1][j] > max) {
					max = B_Dist[u1][j];
					index = j;
				}
			}
			if (index == -1) { logFile << "Error"; } 
			if (distance < max) {
				B_Dist[u1][index] = distance;
				B_Index[u1][index] = u2;
				B_IsNew[u1][index] = flag;
				return 1;
			}
			else { return 0; }
		}
	};
	/**
	 * Main Loop of the Algorithm
	 */
	bool iterate = true;
	while (iterate) {
		/**
		 * Create "New" for each Datapoint
		 */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < K; ++j) {
				if (float(rand() % 100) / 100 < sampleRate) {
					if (B_IsNew[i][j] == 1) {
						New_Index[i].push_back(B_Index[i][j]);
						B_IsNew[i][j] = 0;
					}
				}
			}
		}
		/**
		 * Create "New'"(or REVERSE("New")) for each Datapoint
		 */
		for (int i = 0; i < N; ++i) {
			for (size_t j = 0; j < New_Index[i].size(); ++j) {
				Reverse_New_Index[New_Index[i][j]].push_back(i);
			}
		}
		/**
		 * Random Sampling from "New'"
		 */
		for (int i = 0; i < N; ++i) {
			for (size_t j = 0; j < Reverse_New_Index[i].size(); ++j) {
				if (float(rand() % 100) / 100 < sampleRate) {
					Sampled_Reverse_New_Index[i].push_back(Reverse_New_Index[i][j]);
				}
			}
		}
		/**
		 * "New"= "New" U SAMPLE("New'", pK)
		 */
		for (int i = 0; i < N; ++i) {
			for (size_t j = 0; j < New_Index[i].size(); ++j) {
				New_Final_List[i].push_back(New_Index[i][j]);
			}
			for (size_t j = 0; j < Sampled_Reverse_New_Index[i].size(); ++j) {
				New_Final_List[i].push_back(Sampled_Reverse_New_Index[i][j]);
			}
		}
		/**
		 * c=c+UPDATENN(B[u1],<u2,l,true>)
		 */
		int c_criteria = 0;
		for (int i = 0; i < N; ++i) {
			for (it = New_Final_List[i].begin(); it != New_Final_List[i].end(); it++) {
				it_temp = it;
				advance(it_temp, 1);
				for (it2 = it_temp; it2 != New_Final_List[i].end(); it2++) {
					if (*it != *it2) {
						/**
						 * computes spatial distance between two points
						 * @param  a and b represents the indice of two points in dataset,
						 * data represents input dataPoints read from filePath, Dim is #columns in dataset
						 * @return spatial distance between points a and b
						 */
						double dist = 0;
						for (int i = 0; i < Dim; ++i) {
							dist += pow((dataPoints[*it][i] - dataPoints[*it2][i]), 2);
						}
						double dista = sqrt(dist);
						if (dista < epsilon) {logFile << "Found Duplicate Data for Points "<< *it << " and " << *it2; }

						c_criteria += UpdateNN(*it, *it2, dista, 1);
						c_criteria += UpdateNN(*it2, *it, dista, 1);
					}
				}
			}
		}
		logFile << "c_criteria = " << c_criteria << " With Threshold Convergence of " << convThreshold << endl;
		if (c_criteria < convThreshold) { iterate = 0; }
		/**
		 * Clear the contents of the used data structures
		 */
		for (int i = 0; i < N; ++i) {
			New_Index[i].clear();
			Reverse_New_Index[i].clear();
			Sampled_Reverse_New_Index[i].clear();
			New_Final_List[i].clear();
		}
	}
	logFile<<"------------Ending of K-NN Solution------------"<<endl;
	return;
}


