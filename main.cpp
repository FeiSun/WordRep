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

void help()
{

	cout << "Word vector estimation toolkit v 0.1" << endl << endl;
	cout << "Options:" << endl;
	cout << "Parameters for training:" << endl;
	cout << "\t-train <file>" << endl;
	cout << "\t\tUse text data from <file> to train the model" << endl;
	cout << "\t-word_output <file>" << endl;
	cout << "\t\tUse <file> to save the resulting word vectors" << endl;
	cout << "\t-doc_output <file>" << endl;
	cout << "\t\tUse <file> to save the resulting doc vectors" << endl;
	cout << "\t-word_size <int>"<< endl;
	cout << "\t\tSet size of word vectors; default is 100"<< endl;
	cout << "\t-doc_size <int>"<< endl;
	cout << "\t\tSet size of doc vectors; default is 100"<< endl;
	cout << "\t-window <int>"<< endl;
	cout << "\t\tSet max skip length between words; default is 5"<< endl;
	cout << "\t-subsample <float>"<< endl;
	cout << "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data"<< endl;
	cout << "\t\twill be randomly down-sampled; default is 1e-4, useful range is (0, 1e-5)"<< endl;
	cout << "\t-negative <int>" << endl;
	cout << "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)" << endl;
	cout << "\t-threads <int>" << endl;
	cout << "\t\tUse <int> threads (default 12)" << endl;
	cout << "\t-iter <int>" << endl;
	cout << "\t\tRun more training iterations; default is 5;" << endl;
	cout << "\t-min-count <int>" << endl;
	cout << "\t\tThis will discard words that appear less than <int> times; default is 5" << endl;
	cout << "\t-alpha <float>" << endl;
	cout << "\t\tSet the starting learning rate; default is 0.025 for HDC and 0.05 for PDC" << endl;
	cout << "\t-binary <int>" << endl;
	cout << "\t\tSave the resulting vectors in binary moded; default is 0 (off)" << endl;
	cout << "\t-save-vocab <file>" << endl;
	cout << "\t\tThe vocabulary will be saved to <file>" << endl;
	cout << "\t-model <string>" << endl;
	cout << "\t\tThe model; default is Parallel Document Context model(pdc) (use hdc for Hierarchical Document Context model)" << endl;
	cout << "\nExamples:" << endl;
	cout << "./w2v -train data.txt -word_output vec.txt -size 200 -window 5 -subsample 1e-4 -negative 5 -model pdc -binary 0 -iter 5" << endl;
}

int ArgPos(char *str, int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
		if (!strcmp(str, argv[i])) {
			if (i == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return i;
		}
		return -1;
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int i = 0;
	if (argc == 1)
	{
		help();
		return 0;
	}

	string input_file = "";
	string w_output_file = "";
	string d_output_file = "";
	string save_vocab_file = "";
	string read_vocab_file = "";
	string model = "pdc";
	int table_size = 100000000;
	int word_dim = 100;
	int doc_dim = 100;
	float init_alpha = 0.025f;
	int window = 5;
	float subsample_threshold = 1e-4;
	float min_alpha = init_alpha * 0.0001;
	int negative = 5;
	int num_threads = 12;
	int iter = 5;
	int min_count = 5;
	bool binary = false;

	if ((i = ArgPos((char *)"-word_size", argc, argv)) > 0)
		word_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-doc_size", argc, argv)) > 0)
		doc_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
		input_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
		save_vocab_file = std::string(argv[i + 1]);
	/*if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
		read_vocab_file = std::string(argv[i + 1]);*/
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
		if (atoi(argv[i + 1]) == 1)
			binary = true;
	if ((i = ArgPos((char *)"-model", argc, argv)) > 0)
		model = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
		init_alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-word_output", argc, argv)) > 0)
		w_output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-doc_output", argc, argv)) > 0)
		d_output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-subsample", argc, argv)) > 0)
		subsample_threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
		num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
		iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);

	WordRep w2v(iter, window, min_count, table_size, word_dim, doc_dim, negative, subsample_threshold,
		init_alpha, min_alpha, num_threads, model, binary);

	omp_set_num_threads(num_threads);

	w2v.train(input_file);
	if(w_output_file != "")
		w2v.save_word2vec(w_output_file, w2v.W);
	if(d_output_file != "")
		w2v.save_doc2vec(d_output_file, w2v.D);
	if(save_vocab_file != "")
		w2v.save_vocab(save_vocab_file);

}