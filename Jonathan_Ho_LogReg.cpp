// Jonathan Ho
// CS 4375.003
// Component 4 - Logistic Regression

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <cstdlib>
#include <chrono>
using namespace std;

// Multiply data matrix and weights
vector<double> mult_data_weights(vector<vector<double>> v1, vector<double> v2) {
	vector<double> result;
	double sum = 0.0;

	// Assuming matrices are compatable to multiply
	for (size_t x = 0; x < v1[0].size(); x++) {
		for (size_t y = 0; y < v2.size(); y++) {
			sum += v1[y].at(x) * v2.at(y);
		}
		result.push_back(sum);
		sum = 0.0;
	}

	return result;
}

// Sigmoid function
vector<double> sigmoid(vector<double> s) {
	vector<double> sigmoid_values;
	double sig_double = 0.0;

	for (size_t q = 0; q < s.size(); q++) {
		sig_double = 1.0 / (1.0 + exp(-s.at(q)));
		sigmoid_values.push_back(sig_double);
	}

	return sigmoid_values;
}

// Subtract two matrices
vector<double> sub_matrix(vector<double> f1, vector<double> f2) {
	vector<double> diff_matrix;
	size_t z = 0;

	while (z < f1.size()) {
		diff_matrix.push_back(f1.at(z) - f2.at(z));
		z++;
	}

	return diff_matrix;
}

// Updating the weights matrix after each iteration
vector<double> update_weights(vector<double> wght, double l_rate, vector<vector<double>> data, vector<double> err) {
	vector<double> update;
	double final_sum = 0.0;

	// Transpose multiplication
	for (size_t a = 0; a < data.size(); a++) {
		for (size_t b = 0; b < err.size(); b++) {
			final_sum += data[a].at(b) * err.at(b);
		}

		update.push_back(final_sum);
		final_sum = 0.0;
	}

	// Multiplying learning rate to each value and adding to weight
	for (size_t c = 0; c < update.size(); c++) {
		update.at(c) *= l_rate;
		update.at(c) += wght.at(c);
	}

	return update;
}

// Gradient descent
vector<double> grad_desc(vector<vector<double>> dm, vector<double> w, vector<double> l, double lr) {
	vector<vector<double>> prob_vector(1);
	vector<double> error;

	for (int i = 0; i < 750; i++) {
		prob_vector.at(0) = sigmoid(mult_data_weights(dm, w));
		error = sub_matrix(l, prob_vector[0]);
		w = update_weights(w, lr, dm, error);
	}

	return w;
}

// Log odds
vector<double> log_odds(vector<double> p) {
	vector<double> updated_odds;

	for (size_t h = 0; h < p.size(); h++) {
		updated_odds.push_back(exp(p.at(h)) / (1 + exp(p.at(h))));
	}

	return updated_odds;
}

// Find out number of TP, FP, FN, and TN
vector<double> if_else(vector<double> prob, double threshold) {
	vector<double> truth_table;

	for (size_t t = 0; t < prob.size(); t++) {
		if (prob.at(t) > threshold) {
			truth_table.push_back(1);
		}
		else
		{
			truth_table.push_back(0);
		}
	}

	return truth_table;
}

// Calculate accuracy
double calc_acc(vector<double> pred_acc, vector<double> real_acc) {
	int true_num = 0;

	for (size_t y = 0; y < pred_acc.size(); y++) {
		if (pred_acc.at(y) == real_acc.at(y)) {
			true_num++;
		}
	}

	return static_cast<double>(true_num) / pred_acc.size();
}

// Calculate sensitivity
double calc_sens(vector<double> pred_sens, vector<double> real_sens) {
	int true_pos = 0;
	int false_neg = 0;

	for (size_t z = 0; z < pred_sens.size(); z++) {
		if (pred_sens.at(z) == 0 && real_sens.at(z) == 0) {
			true_pos++;
		}

		if (pred_sens.at(z) == 1 && real_sens.at(z) == 0) {
			false_neg++;
		}
	}

	return static_cast<double>(true_pos) / (static_cast<double>(true_pos) + false_neg);
}

// Calculate specificity
double calc_spec(vector<double> pred_spec, vector<double> real_spec) {
	int true_neg = 0;
	int false_pos = 0;

	for (size_t w = 0; w < pred_spec.size(); w++) {
		if (pred_spec.at(w) == 1 && real_spec.at(w) == 1) {
			true_neg++;
		}

		if (pred_spec.at(w) == 0 && real_spec.at(w) == 1) {
			false_pos++;
		}
	}

	return static_cast<double>(true_neg) / (static_cast<double>(true_neg) + false_pos);
}

int main(int argc, char** argv) {

	ifstream inFS; // Input file stream
	string line;
	string index_in, pclass_in, survived_in, sex_in, age_in;
	const int MAX_LEN = 1100;
	vector<double> index(MAX_LEN);
	vector<double> pclass(MAX_LEN);
	vector<double> survived(MAX_LEN);
	vector<double> sex(MAX_LEN);
	vector<double> age(MAX_LEN);

	// Try to open file
	cout << "Opening file titanic_project.csv." << endl;

	inFS.open("titanic_project.csv");
	if (!inFS.is_open()) {
		cout << "Coult not open file titanic_project.csv." << endl;
		return 1; // 1 indicates error
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	// echo heading
	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {

		getline(inFS, index_in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, sex_in, ',');
		getline(inFS, age_in, '\n');

		// If the end of the csv file is reached
		if (index_in.compare("") == 0) {
			break;
		}

		index.at(numObservations) = stof(index_in);
		pclass.at(numObservations) = stof(pclass_in);
		survived.at(numObservations) = stof(survived_in);
		sex.at(numObservations) = stof(sex_in);
		age.at(numObservations) = stof(age_in);

		numObservations++;
	}

	index.resize(numObservations);
	pclass.resize(numObservations);
	survived.resize(numObservations);
	sex.resize(numObservations);
	age.resize(numObservations);

	cout << "Closing file titanic_project.cvs." << endl;
	inFS.close(); // Done with file, so close it

	cout << "Number of records: " << numObservations << endl;

	////////// LOGISTICAL REGRESSION //////////
	
	// Time start
	chrono::steady_clock::time_point start = chrono::steady_clock::now();

	// Clean this up later, also check TEST_LEN value again later
	const int TRAIN_LEN = 800;
	const int TEST_LEN = 246;
	vector<double> weights = { 1.0, 1.0 };
	vector<vector<double>> data_matrix; 
	vector<double> labels;
	vector<double> train_sex;
	vector<vector<double>> test_matrix;
	vector<double> test_labels;
	vector<double> test_sex;
	vector<double> predicted;
	vector<double> probabilities;
	vector<double> predictions;
	vector<double> ones(TRAIN_LEN);
	double learning_rate = 0.001; // Learning rate

	// Create the training and test data
	copy(survived.begin(), survived.begin() + TRAIN_LEN, back_inserter(labels));
	copy(sex.begin(), sex.begin() + TRAIN_LEN, back_inserter(train_sex)); 
	copy(survived.begin() + TRAIN_LEN, survived.end(), back_inserter(test_labels)); 
	copy(sex.begin() + TRAIN_LEN, sex.end(), back_inserter(test_sex)); 

	// Fill data_matrix
	fill_n(ones.begin(), ones.size(), 1.0);

	// First column is ones, second is sex
	data_matrix.push_back(ones);
	data_matrix.push_back(train_sex);

	// Gradient descent
	weights = grad_desc(data_matrix, weights, labels, learning_rate);

	// Coefficients for Intercept and Sex respectively
	cout << "\nCoefficients: " << endl;
	for (size_t g = 0; g < weights.size(); g++) {
		cout << weights.at(g) << endl;
	}
	cout << endl;

	// Calculating accuracy, sensitivity, and specificity

	// Update ones matrix size and fill test_matrix
	ones.resize(TEST_LEN);
	test_matrix.push_back(ones);
	test_matrix.push_back(test_sex);

	// Calculate predicted
	predicted = mult_data_weights(test_matrix, weights);
	probabilities = log_odds(predicted);
	predictions = if_else(probabilities, 0.5);

	// Calculate accuracy, sensitivity, and specificity with 0, 0 being TP and 1, 1 being TN
	cout << "Accuracy: " << calc_acc(predictions, test_labels) << endl;
	cout << "Sensitivity: " << calc_sens(predictions, test_labels) << endl;
	cout << "Specificity: " << calc_spec(predictions, test_labels) << endl;

	// Time end
	chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "\nThe algorithm runtime is " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";

	cout << "\nProgram terminated.";

	return 0;
}