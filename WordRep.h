//  Copyright (c) 2015 Fei Sun. All Rights Reserved.
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//  For more information, bug reports, fixes, contact:
//  Fei Sun (ofey.sunfei@gmail.com)
//  http://ofey.me/projects/wordrep/

#pragma once
#include <vector>
#include <list>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <limits>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "Word.h"

using namespace std;
using namespace Eigen;

#define EIGEN_NO_DEBUG
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;


class WordRep
{
public:
	int iter;
	int window;
	int min_count;
	int table_size;
	int word_dim;
	int doc_dim ;
	int negative;
	float subsample_threshold;
	float init_alpha;
	float min_alpha;
	float ep;
	int num_threads;
	long doc_num;
	long long total_words;

	bool binary;

	string model;

	vector<Word *> vocab;
	unordered_map<string, WordP> vocab_hash;
	vector<size_t> table;
	std::uniform_int_distribution<int> distribution_table;
	std::uniform_real_distribution<float> uni_dis;
	std::uniform_int_distribution<int> distribution_window;

	RMatrixXf W, W_his, C, C_his, D, D_his;

	std::random_device rd;
	std::mt19937 generator;

public:
	WordRep(int iter=5, int window=10, int min_count=5, int table_size=100000000, int word_dim=100,
		int doc_dim=100, int negative=0, float subsample_threshold=0.0001,float init_alpha=0.025,
		float min_alpha=1e-6, int num_threads=1, string model="pdc", bool binary=false);
	~WordRep(void);

	vector<vector<string>> line_docs(string filename);
	void reduce_vocab();
	void make_table(vector<size_t>& table, vector<Word *>& vocab);
	void precalc_sampling();
	void build_vocab(vector<vector<string>>& docs);
	void build_vocab(string filename);
	void init_weights(size_t vocab_size, size_t doc_size);
	vector<vector<Word *>> build_docs(vector<vector<string>>& data);
	vector<vector<Word *>> build_docs(string filename);

	void negative_sampling(float alpha, Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad,
		                   RMatrixXf& target_matrix, RMatrixXf& target_matrix_his);

	void train_hdc(vector<vector<Word *>>& docs);
	void train_pdc(vector<vector<Word *>>& docs);

	void train(string filename);

	void save_vocab(string vocab_filename);
	void save_word2vec(string filename, const RMatrixXf& data, bool binary=false);
	void save_doc2vec(string filename, const RMatrixXf& data);
};

