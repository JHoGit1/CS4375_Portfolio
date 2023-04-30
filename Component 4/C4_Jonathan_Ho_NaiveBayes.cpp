// Jonathan Ho
// CS 4375.003
// Component 4 - Naive Bayes

#define _USE_MATH_DEFINES
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

// Returns the min and max of the data set after it is sorted
vector<double> find_range(vector<double> data) {
	vector<double> rangeData;
	double min_num = 0;
	double max_num = 0;

	sort(data.begin(), data.end());

	rangeData.push_back(data[0]);
	rangeData.push_back(data[data.size() - 1]);

	// If there is more than 2 elements, fill it up to that size with values up to the max number
	if (rangeData[1] - rangeData[0] > 1) {
		min_num = rangeData[0];
		max_num = rangeData[1];
		rangeData.clear();
		for (int f = 0; f < max_num; f++) {
			rangeData.push_back(min_num);
			min_num += 1.0f;
		}
	}

	return rangeData;
}

// Calculate priors
vector<double> calc_prior(vector<double> prior) {
	vector<double> pri;
	int num_zero = 0;
	int num_one = 0;

	for (size_t a = 0; a < prior.size(); a++) {
		// Find number of 1s and 0s
		if (prior.at(a) == 0) {
			num_zero++;
		}
		else {
			num_one++;
		}
	}

	pri.push_back(static_cast<double>(num_zero) / prior.size());
	pri.push_back(static_cast<double>(num_one) / prior.size());

	return pri;
}

// Calculate number of zeros (no) and ones (yes) of a data set
vector<double> count_yn(vector<double> yn) {
	vector<double> count;
	int n_zero = 0;
	int n_one = 0;

	for (size_t b = 0; b < yn.size(); b++) {
		if (yn.at(b) == 0) {
			n_zero++;
		}
		else {
			n_one++;
		}
	}

	count.push_back(n_zero);
	count.push_back(n_one);

	return count;
}

// Likelihood for quantitative class
vector<vector<double>> lh_quant(vector<double> c1, vector<double> c2, vector<double> factor) {
	vector<vector<double>> total_lh;
	vector<double> lh;
	vector<double> data_range_c1;
	vector<double> data_range_c2;
	int num_match = 0;
	int c1_pos = 0;
	int c2_pos = 0;

	// Find the range of data within each data
	data_range_c1 = find_range(c1);
	data_range_c2 = find_range(c2);

	for (size_t c = 0; c < data_range_c1.size(); c++) {
		for (size_t d = 0; d < data_range_c2.size(); d++) {
			for (size_t e = 0; e < c1.size(); e++) {
				// If the number matches both classes
				if (c1.at(e) == data_range_c1.at(c1_pos) && c2.at(e) == data_range_c2.at(c2_pos)) {
					num_match++;
				}
			}
			// Push the specific number into the vector
			lh.push_back(static_cast<double>(num_match) / factor.at(c));
			c2_pos++;
			num_match = 0;
		}

		// Push the row into a vector
		total_lh.push_back(lh);
		lh.clear();
		c1_pos++;
		c2_pos = 0;
	}

	return total_lh;
}

// Calculate mean of quantitative data
vector<double> nb_mean(vector<double> factor, vector<double> dis, vector<double> counted) {
	vector<double> total_mean;
	double sum = 0.0;
	size_t counter = 0;

	while (counter < counted.size()) {
		for (size_t g = 0; g < dis.size(); g++) {
			if (factor.at(g) == counter) {
				sum += dis.at(g);
			}
		}

		total_mean.push_back(sum / counted.at(counter));
		sum = 0;
		counter++;
	}

	return total_mean;
}

// Calculate variance of quantitative data
vector<double> nb_var(vector<double> fact, vector<double> dis_var, vector<double> num_count, vector<double> mean) {
	vector<double> final_var;
	double sumVar = 0.0;
	size_t pos = 0;

	while (pos < num_count.size()) {
		for (size_t h = 0; h < dis_var.size(); h++) {
			if (fact.at(h) == pos) {
				sumVar += pow(dis_var.at(h) - mean.at(pos), 2.0);
			}
		}

		final_var.push_back(sumVar / (num_count.at(pos) - 1)); // - 1 is for sample
		sumVar = 0;
		pos++;
	}

	return final_var;
}

// Calculating probability density
double calc_age_lh(double v, double mean_v, double var_v) {
	return (1 / sqrt(2 * M_PI * var_v) * exp(-((pow(v - mean_v, 2.0)) / (2 * var_v))));
}

// Calculate raw probabilities
vector<double> calc_raw_prob(double surv[], double per[], double age_s, double age_p, vector<double> prior) {
	vector<double> raw_list;
	double num_s = 0.0;
	double num_p = 0.0;
	double denominator = 0.0;

	num_s = surv[0] * surv[1] * prior.at(1) * age_s;
	num_p = per[0] * per[1] * prior.at(0) * age_p;
	denominator = num_s + num_p;
	raw_list.push_back(num_s / denominator);
	raw_list.push_back(num_p / denominator);

	return raw_list;
}

// Create a confusion matrix
vector<int> confusionMatrix(vector<vector<double>> p, vector<double> t) {
	vector<int> cm;
	int val = 0;
	int TP = 0;
	int FP = 0;
	int FN = 0;
	int TN = 0;

	for (size_t i = 0; i < p.size(); i++) {
		// 1 is perished, 0 is survived
		if (p[i].at(0) < p[i].at(1)) {
			val = 0;
		}
		else {
			val = 1;
		}

		if (val == t.at(i) && val == 0) {
			TP++;
		}
		else if (val == t.at(i) && val == 1) {
			TN++;
		}
		else if (val != t.at(i) && val == 0) {
			FP++;
		}
		else {
			FN++;
		}
	}

	// Push in a table starting from top and reading from left to right
	cm.push_back(TP);
	cm.push_back(FP);
	cm.push_back(FN);
	cm.push_back(TN);

	return cm;
}

// Calculate accuracy
double acc_calc(vector<int> m) {
	return (static_cast<double>(m[0]) + m[3]) / (static_cast<double>(m[0]) + m[1] + m[2] + m[3]);
}

// Calculate sensitivity
double sens_calc(vector<int> m) {
	return (static_cast<double>(m[0]) / (static_cast<double>(m[0]) + m[2]));
}

// Calculate specificity
double spec_calc(vector<int> m) {
	return (static_cast<double>(m[3]) / (static_cast<double>(m[3]) + m[1]));
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

	cout << "Number of records: " << numObservations << endl << endl;

	////////// NAIVE BAYES ////////// 

	// Time start
	chrono::steady_clock::time_point start = chrono::steady_clock::now();

	// Instantiation
	const int TRAIN_LEN = 800;
	const int TEST_LEN = 246;
	vector<double> test_pclass;
	vector<double> test_sex;
	vector<double> test_age;
	vector<double> test_survived;
	vector<double> apriori;
	vector<double> count_survived;
	vector<vector<double>> lh_pclass;
	vector<vector<double>> lh_sex;
	vector<double> age_mean;
	vector<double> age_var;
	vector<vector<double>> raw_prob;
	vector<int> c_matrix;

	// Functions to fill the test of the classes
	copy(pclass.begin() + TRAIN_LEN, pclass.end(), back_inserter(test_pclass));
	copy(sex.begin() + TRAIN_LEN, sex.end(), back_inserter(test_sex));
	copy(age.begin() + TRAIN_LEN, age.end(), back_inserter(test_age));
	copy(survived.begin() + TRAIN_LEN, survived.end(), back_inserter(test_survived));

	// Calculating prior values
	apriori = calc_prior(survived);

	cout << "Prior probabilities 0 = perished, 1 = survived" << endl;
	for (size_t aa = 0; aa < apriori.size(); aa++) {
		cout << "[" << aa << "] " << apriori.at(aa) << endl;
	}
	cout << endl;
	
	// Get count of not survived and survived
	count_survived = count_yn(survived);

	// Get likelihoods of qualitative classes. Factor as the 3rd arguement
	lh_pclass = lh_quant(survived, pclass, count_survived);
	lh_sex = lh_quant(survived, sex, count_survived);

	cout << "For p(survived|pclass):" << endl;
	for (size_t xx = 0; xx < lh_pclass.size(); xx++) {
		for (size_t yy = 0; yy < lh_pclass[xx].size(); yy++) {
			cout << "[" << xx << ", " << yy << "] " << lh_pclass[xx].at(yy) << endl;
		}
	}

	cout << "\nFor p(survived|sex):" << endl;
	for (size_t xy = 0; xy < lh_sex.size(); xy++) {
		for (size_t yx = 0; yx < lh_sex[xy].size(); yx++) {
			cout << "[" << xy << ", " << yx << "] " << lh_sex[xy].at(yx) << endl;
		}
	}
	cout << "\n";

	// Likelihood for Condinous Data (quantitative)

	// Get mean and var
	age_mean = nb_mean(survived, age, count_survived);
	age_var = nb_var(survived, age, count_survived, age_mean);

	// Get all the raw probabilities of the test data
	for (int x = 0; x < TEST_LEN; x++) {
		// Array created for each aspect of calculating the raw probability
		double test_obs[3] = { test_pclass.at(x), test_sex.at(x), test_age.at(x) };
		double lh_s[2] = { lh_pclass[1].at(static_cast<unsigned int>(test_obs[0] - 1.0)), lh_sex[1].at(static_cast<unsigned int>(test_obs[1])) }; // - 1 for index of pclass
		double lh_p[2] = { lh_pclass[0].at(static_cast<unsigned int>(test_obs[0] - 1.0)), lh_sex[0].at(static_cast<unsigned int>(test_obs[1])) }; // - 1 for index of pclass
		double lh_age_s = calc_age_lh(test_obs[2], age_mean.at(1), age_var.at(1));
		double lh_age_p = calc_age_lh(test_obs[2], age_mean.at(0), age_var.at(0));

		raw_prob.push_back(calc_raw_prob(lh_s, lh_p, lh_age_s, lh_age_p, apriori));
	}
	
	// Print the first 5 probabilities of test (index 1 is perished, index 0 is survived)
	for (size_t pos_vec = 0; pos_vec < 5; pos_vec++) {
		cout << "[" << pos_vec << "] " << raw_prob[pos_vec].at(1) << " " << raw_prob[pos_vec].at(0) << endl;
	}

	// Creating confusion matrix and calculating accuracy, sensitivity, and specificity
	c_matrix = confusionMatrix(raw_prob, test_survived);
	cout << "\nAccuracy: " << acc_calc(c_matrix) << endl;
	cout << "Sensitivity: " << sens_calc(c_matrix) << endl;
	cout << "Specificity: " << spec_calc(c_matrix) << endl;
	
	// Time end
	chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "\nThe algorithm runtime is " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";

	cout << "\nProgram terminated.";

	return 0;
}