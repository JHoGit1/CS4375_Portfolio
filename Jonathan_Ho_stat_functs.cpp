// Jonathan Ho
// CS 4375.003

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
using namespace std;

// Adds all the values within the given data set
double sum(vector<double> data) {
	double sumTotal = 0;

	// size_t to get rid of warning of .size() conversion
	for (size_t i = 0; i < data.size(); i++)
	{
		sumTotal += data[i];
	}

	return sumTotal;
}

// Finds the average of the values within the given data set
double mean(vector<double> data) {
	return (sum(data) / data.size());
}

// Find the "middle" of the data set after it is sorted
double median(vector<double> data) {
	int mid = 0;

	// Sort the data into Ascending order
	sort(data.begin(), data.end());

	mid = static_cast<int>(ceil(data.size() / 2));

	// Check if there is an even or odd number of data
	if (data.size() % 2 == 0) {
		return ((data[mid] + data[mid - 1]) / 2);
	}
	else {
		return data[mid];
	}
}

// Returns the min and max of the data set after it is sorted
// Can also be stored into another vector
vector<double> range(vector<double> data) {
	vector<double> rangeData;

	sort(data.begin(), data.end());

	rangeData.push_back(data[0]);
	rangeData.push_back(data[data.size() - 1]);

	cout << rangeData[0] << " " << rangeData[1] << endl;

	return rangeData;
}

// Function to utilize all the other statistical functions
void print_stats(vector<double> dataList) {
	cout << "SUM: " << sum(dataList) << endl;
	cout << "MEAN: " << mean(dataList) << endl;
	cout << "MEDIAN: " << median(dataList) << endl;
	cout << "RANGE: ";
	range(dataList);
}

// Calculates and returns the covariance of two different data sets
double covar(vector<double> dataOne, vector<double> dataTwo) {
	double covariance = 0.0;

	// Checks if data sets are the same size
	if (dataOne.size() == dataTwo.size()) {
		for (size_t i = 0; i < dataOne.size(); i++) {
			covariance += (dataOne[i] - mean(dataOne)) * (dataTwo[i] - mean(dataTwo));
		}
		
		covariance /= dataOne.size() - 1;
	}
	else {
		cout << "Data set sizes are not equal." << endl;
	}
	
	return covariance;
}

// Calculates and returns the correlation between two different data sets
double cor(vector<double> dataOne, vector<double> dataTwo) {
	double sdOne = 0.0;
	double sdTwo = 0.0;
	double sumOne = 0.0;
	double sumTwo = 0.0;

	if (dataOne.size() != dataTwo.size())
	{
		cout << "Data sizes are not equivalent." << endl;
		return 0.0;
	}

	for (size_t x = 0; x < dataOne.size(); x++) {
		sumOne += pow(dataOne[x] - mean(dataOne), 2.0);
		sumTwo += pow(dataTwo[x] - mean(dataTwo), 2.0);
	}

	sdOne = sqrt(sumOne / (dataOne.size() - 1));
	sdTwo = sqrt(sumTwo / (dataTwo.size() - 1));

	return covar(dataOne, dataTwo) / (sdOne * sdTwo);
}

int main(int argc, char** argv) {

	ifstream inFS; // Input file stream
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	// Try to open file
	cout << "Opening file Boston.csv." << endl;

	inFS.open("Boston.csv");
	if (!inFS.is_open()) {
		cout << "Coult not open file Boston.csv." << endl;
		return 1; // 1 indicates error
	}

	// Can now use inFS stream like cin stream
	// Boston.csv should contain two doubles

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	// echo heading
	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {

		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "new length " << rm.size() << endl;

	cout << "Closing file Boston.cvs." << endl;
	inFS.close(); // Done with file, so close it

	cout << "Number of records: " << numObservations << endl;

	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\nCovariance = " << covar(rm, medv);
	cout << "\nCorrelation = " << cor(rm, medv) << endl;

	cout << "\nProgram terminated.";

	return 0;
}