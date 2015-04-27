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

#ifndef WORD_H
#define WORD_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

using namespace std;

class Word
{
public:
	size_t index;
	size_t count;
	float sample_probability;
	string text;
	Word *left, *right;

	std::vector<size_t> codes;
	std::vector<size_t> points;

public:
	Word(void){};
	Word(size_t index, size_t count, string text, Word *left = nullptr, Word *right = nullptr):
	    index(index), count(count), text(text), left(left), right(right) {}

	~Word(void){};
};

typedef std::shared_ptr<Word> WordP;

#endif