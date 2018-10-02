/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder.language

import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.properties.datetime.DateTime
import com.kotlinnlp.linguisticdescription.sentence.properties.MultiWords

/**
 * The sentence used as input of the [com.kotlinnlp.lssencoder.LSSEncoder].
 *
 * @property tokens the list of tokens of the sentence
 * @property multiWords the list of multi-words expressions recognized in the sentence (can be empty)
 * @property dateTimes the list of date-times expressions recognized in the sentence (can be empty)
 */
class ParsingSentence(
  override val tokens: List<ParsingToken>,
  override val multiWords: List<MultiWords> = emptyList(),
  override val dateTimes: List<DateTime> = emptyList()
) : MorphoSentence<ParsingToken>, SentenceIdentificable<ParsingToken>()
