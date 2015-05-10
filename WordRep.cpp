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

#include "WordRep.h"

WordRep::WordRep(int iter, int window, int min_count, int table_size, int word_dim, int doc_dim, int negative,
	float subsample_threshold, float init_alpha, float min_alpha, int num_threads, string model, bool binary):
iter(iter),  window(window), min_count(min_count), table_size(table_size), word_dim(word_dim), doc_dim(doc_dim),
	negative(negative), subsample_threshold(subsample_threshold), init_alpha(init_alpha),
	min_alpha(min_alpha), num_threads(num_threads), model(model), binary(binary),
	generator(rd()), distribution_table(0, table_size - 1),
	uni_dis(0.0, 1.0), distribution_window(0, window < 1 ? 0 : window - 1)
{
	doc_num = 0;
	total_words = 0;
	ep = numeric_limits<float>::epsilon();
} 

WordRep::~WordRep(void)
{
}

inline bool comp(Word *w1, Word *w2)
{
	return w1->count > w2->count;
}

vector<vector<string>> WordRep::line_docs(string filename)
{
	vector<vector<string>> docs;
	ifstream in(filename);
	string s;

	while (std::getline(in, s))
	{
		istringstream iss(s);
		docs.emplace_back(istream_iterator<string>{iss}, istream_iterator<string>{});
	}
	return std::move(docs);
}

void WordRep::make_table(vector<size_t>& table, vector<Word *>& vocab)
{
	table.resize(table_size);
	size_t vocab_size = vocab.size();
	float power = 0.75f;
	float train_words_pow = 0.0f;

	vector<float> word_range(vocab.size());
	for(size_t i = 0; i != vocab_size; ++i)
	{
		word_range[i] = pow((float)vocab[i]->count, power);
		train_words_pow += word_range[i];
	}

	size_t idx = 0;
	float d1 = word_range[idx] / train_words_pow;
	float scope = table_size * d1;
	for(int i = 0; i < table_size; ++i)
	{
		table[i] = idx;
		if(i > scope && idx < vocab_size - 1)
		{
			d1 += word_range[++idx] / train_words_pow;
			scope = table_size * d1;
		}
		else if(idx == vocab_size - 1)
		{
			for(; i < table_size; ++i)
				table[i] = idx;
			break;
		}
	}
}

void WordRep::precalc_sampling()
{
	size_t vocab_size = vocab.size();
	size_t word_count = 0;

	float threshold_count  = subsample_threshold * total_words;

	if(subsample_threshold > 0)
		for(auto& w: vocab)
			w->sample_probability = std::min(float((sqrt(w->count / threshold_count) + 1) * threshold_count / w->count), (float)1.0);
	else
		for(auto& w: vocab)
			w->sample_probability = 1.0;
}

void WordRep::build_vocab(vector<vector<string>>& docs)
{
	doc_num = docs.size();
	unordered_map<string, int> word_cn;

	for(auto& doc: docs)
	{
		for(auto& w: doc)
			if(word_cn.count(w) > 0)
				word_cn[w]++;
			else
				word_cn[w] = 1;
	}

	//ignore words apper less than min_count
	total_words = 0;
	for(auto kv: word_cn)
	{
		if(kv.second < min_count)
			continue;

		Word *w = new Word(0, kv.second,  kv.first);
		vocab.push_back(w);
		vocab_hash[w->text] = WordP(w);
		total_words += kv.second;
	}

	//update word index
	size_t vocab_size = this->vocab.size();
	sort(this->vocab.begin(), this->vocab.end(), comp);
	for(size_t i = 0; i < vocab_size; ++i)
	{
		this->vocab[i]->index = i;
	}

	make_table(this->table, this->vocab);
	precalc_sampling();
}

void WordRep::build_vocab(string filename)
{
	ifstream in(filename);
	string s, w;
	unordered_map<string, size_t> word_cn;

	while (std::getline(in, s))
	{
		doc_num++;
		istringstream iss(s);
		while (iss >> w)
		{
			if(word_cn.count(w) > 0)
				word_cn[w]++;
			else
				word_cn[w] = 1;
		}
	}
	in.close();
	//ignore words apper less than min_count
	for(auto kv: word_cn)
	{
		if(kv.second < min_count)
			continue;

		Word *w = new Word(0, kv.second,  kv.first);
		vocab.push_back(w);
		vocab_hash[w->text] = WordP(w);
		total_words += kv.second;
	}
	//update word index
	size_t vocab_size = vocab.size();
	sort(vocab.begin(), vocab.end(), comp);
	for(size_t i = 0; i < vocab_size; ++i)
	{
		vocab[i]->index = i;
	}

	make_table(this->table, this->vocab);
	precalc_sampling();
}

void WordRep::init_weights(size_t vocab_size, size_t doc_size)
{
	std::uniform_real_distribution<float> distribution(-0.5, 0.5);
	auto uniform = [&] (int) {return distribution(generator);};

	D = RMatrixXf::NullaryExpr(doc_size, doc_dim, uniform) / (float)doc_dim;
	W = RMatrixXf::NullaryExpr(vocab_size, word_dim, uniform) / (float)word_dim;
	#ifdef ADAGRAD
	D_his = RMatrixXf::Zero(doc_size, doc_dim);
	W_his = RMatrixXf::Zero(vocab_size, doc_dim);
	#endif


	C = RMatrixXf::NullaryExpr(vocab_size, word_dim, uniform) / (float)word_dim;;
	#ifdef ADAGRAD
	C_his = RMatrixXf::Zero(vocab_size, doc_dim);
    #endif
}

vector<vector<Word *>> WordRep::build_docs(vector<vector<string>>& data)
{
	vector<vector<Word *>> docs;
	for(auto& sentence: data)
	{
		vector<Word *> doc;

		for(auto text: sentence)
		{
			auto it = vocab_hash.find(text);
			if (it == vocab_hash.end()) continue;
			Word *word = it->second.get();

			doc.push_back(word);
		}
		docs.push_back(std::move(doc));
	}

	return std::move(docs);
}

vector<vector<Word *>> WordRep::build_docs(string filename)
{
	ifstream in(filename);
	string s, w;

	vector<vector<Word *>> docs;

	while (std::getline(in, s))
	{
		vector<Word *> doc;
		istringstream iss(s);

		while (iss >> w)
		{
			auto it = vocab_hash.find(w);
			if (it == vocab_hash.end()) continue;
			Word *word = it->second.get();

			doc.push_back(word);
		}
		docs.push_back(std::move(doc));
	}
	in.close();

	return std::move(docs);
}

void WordRep::negative_sampling(float alpha, Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad, 
	                            RMatrixXf& target_matrix, RMatrixXf& target_matrix_his)
{
	unordered_map<size_t, uint8_t> targets;
	for (int i = 0; i < negative; ++i)
		targets[table[distribution_table(generator)]] = 0;

	targets[predict_word->index] = 1;

	for (auto it: targets)
	{
		float f = target_matrix.row(it.first).dot(project_rep);
		f = 1.0 / (1.0 + exp(-f));
		float g = it.second - f;

		project_grad += g * target_matrix.row(it.first);
		RowVectorXf l2_grad = g * project_rep;

		#ifdef ADAGRAD
		target_matrix_his.row(it.first) += l2_grad.cwiseProduct(l2_grad);
		RowVectorXf temp = target_matrix_his.row(it.first).cwiseSqrt().array() + ep;
		target_matrix.row(it.first) += alpha * l2_grad.cwiseQuotient(temp);
		#else
		target_matrix.row(it.first) += alpha * l2_grad;
		#endif 
	}
}


void WordRep::train_hdc(vector<vector<Word *>>& docs)
{
	vector<long> sample_idx(docs.size());
	std::iota(sample_idx.begin(), sample_idx.end(), 0);

	long long wn = 0;
	float alpha = init_alpha;

	for(int it = 0; it < iter; ++it)
	{
		cout << "iter:" << it <<endl;
		std::shuffle(sample_idx.begin(), sample_idx.end(), generator);

        #pragma omp parallel for
		for(int i = 0; i < doc_num; ++i)
		{
			if(i % 10 == 0)
			{
				alpha = std::max(min_alpha, float(init_alpha * (1.0 - 1.0 / iter * wn / total_words)));
				#ifdef DEBUG
				printf("\ralpha: %f  Progress: %f%%", alpha, 100.0 / iter * wn / total_words);
				std::fflush(stdout);
				#endif
			}

			long doc_id = sample_idx[i];
			vector<Word *>& doc = docs[doc_id];
			int doc_len = doc.size();
			RowVectorXf neu1_grad = RowVectorXf::Zero(word_dim);

			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				if(current_word->sample_probability < uni_dis(generator))
					continue;

				int reduced_window = distribution_window(generator);
				int index_begin = max(0, j - window + reduced_window);
				int index_end = min((int)doc_len, j + window + 1 - reduced_window);
				
				//paradigmatic
				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;

					neu1_grad.setZero();

					RowVectorXf neu1 = W.row(current_word->index);
					negative_sampling(alpha, doc[m], neu1, neu1_grad, C, C_his);

					#ifdef ADAGRAD
					W_his.row(current_word->index) += neu1_grad.cwiseProduct(neu1_grad);
					RowVectorXf temp = W_his.row(current_word->index).cwiseSqrt().array() + ep;
					W.row(current_word->index) += alpha * neu1_grad.cwiseQuotient(temp);
					#else
					W.row(current_word->index) += alpha * neu1_grad;
					#endif 
				}

				//syntagmatic
				RowVectorXf neu1 = D.row(doc_id);
				neu1_grad.setZero();
				negative_sampling(alpha, current_word, neu1, neu1_grad, W, W_his);

				#ifdef ADAGRAD
				D_his.row(doc_id) += neu1_grad.cwiseProduct(neu1_grad);
				RowVectorXf temp = D_his.row(doc_id).cwiseSqrt().array() + ep;
				D.row(doc_id) += alpha * neu1_grad.cwiseQuotient(temp);
				#else
				D.row(doc_id) += alpha * neu1_grad;
				#endif 
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void WordRep::train_pdc(vector<vector<Word *>>& docs)
{
	vector<int> sample_idx(docs.size());
	std::iota(sample_idx.begin(), sample_idx.end(), 0);

	long long wn = 0;
	float alpha = init_alpha;

	for(int it = 0; it < iter; ++it)
	{
		cout << "iter:" << it <<endl;
		std::shuffle(sample_idx.begin(), sample_idx.end(), generator);

        #pragma omp parallel for
		for(int i = 0; i < docs.size(); ++i)
		{
			if(i % 10 == 0)
			{
				alpha = std::max(min_alpha, float(init_alpha * (1.0 - 1.0 / iter * wn / total_words)));
				#ifdef DEBUG
				printf("\ralpha: %f  Progress: %f%%", alpha, 100.0 / iter * wn / total_words);
				std::fflush(stdout);
				#endif
			}

			long doc_id = sample_idx[i];
			auto doc = docs[doc_id];
			size_t doc_len =  doc.size();

			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				if(current_word->sample_probability < uni_dis(generator))
					continue;

				int reduced_window = distribution_window(generator);
				int index_begin = max(0, j - window + reduced_window);
				int index_end = min((int)doc_len, j + window + 1 - reduced_window);

				RowVectorXf neu1 = RowVectorXf::Zero(word_dim);
				RowVectorXf neu1_grad = RowVectorXf::Zero(word_dim);

				//paradigmatic
				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;
					neu1 += C.row(doc[m]->index);
				}
				if(index_end - index_begin > 1)
					neu1 /= index_end - index_begin - 1;

				negative_sampling(alpha, current_word, neu1, neu1_grad, W, W_his);

				if(index_end - index_begin > 1)
					neu1_grad /= index_end - index_begin - 1;

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;

					#ifdef ADAGRAD
					C_his.row(doc[m]->index) += neu1_grad.cwiseProduct(neu1_grad);
					RowVectorXf temp = C_his.row(doc[m]->index).cwiseSqrt().array() + ep;
					C.row(doc[m]->index) +=  alpha * neu1_grad.cwiseQuotient(temp);
					#else
					C.row(doc[m]->index) += alpha * neu1_grad;
					#endif 
				}

				//syntagmatic
				neu1 = D.row(doc_id);
				neu1_grad.setZero();
				negative_sampling(alpha, current_word, neu1, neu1_grad, W, W_his);

				#ifdef ADAGRAD
				D_his.row(doc_id) += neu1_grad.cwiseProduct(neu1_grad);
				RowVectorXf temp = D_his.row(doc_id).cwiseSqrt().array() + ep;
				D.row(doc_id) += alpha * neu1_grad.cwiseQuotient(temp);
				#else
				D.row(doc_id) += alpha * neu1_grad;
				#endif 
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void WordRep::train(string filename)
{
	build_vocab(filename);
	init_weights(vocab.size(), doc_num);
	vector<vector<Word *>> docs = build_docs(filename);

	if(model == "hdc")
		train_hdc(docs);
	else if(model == "pdc")
		train_pdc(docs);
}

void WordRep::save_vocab(string vocab_filename)
{
	ofstream out(vocab_filename, std::ofstream::out);
	for(auto& v: vocab)
		out << v->index << " " << v->count << " " << v->text << endl;
	out.close();
}

void WordRep::save_word2vec(string filename, const RMatrixXf& data, bool binary)
{
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols);

	if(binary)
	{
		std::ofstream out(filename, std::ios::binary);
		char blank = ' ';
		char enter = '\n'; 
		int size = sizeof(char);
		int r_size = data.cols() * sizeof(RMatrixXf::Scalar);

		RMatrixXf::Index r = data.rows();
		RMatrixXf::Index c = data.cols();
		out.write((char*) &r, sizeof(RMatrixXf::Index));
		out.write(&blank, size);
		out.write((char*) &c, sizeof(RMatrixXf::Index));
		out.write(&enter, size);

		for(auto v: vocab)
		{
			out.write(v->text.c_str(), v->text.size());
			out.write(&blank, size);
			out.write((char*) data.row(v->index).data(), r_size);
			out.write(&enter, size);
		}
		out.close();
	}
	else
	{
		ofstream out(filename);

		out << data.rows() << " " << data.cols() << std::endl;

		for(auto v: vocab)
		{
			out << v->text << " " << data.row(v->index).format(CommaInitFmt) << endl;
		}
		out.close();
	}
}

void WordRep::save_doc2vec(string filename, const RMatrixXf& data)
{
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols);
	ofstream out(filename, std::ofstream::out);

	out << data.rows() << " " << data.cols() << std::endl << data.format(CommaInitFmt);
	out.close();
}