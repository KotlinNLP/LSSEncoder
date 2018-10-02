/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder.language

import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.sentence.token.*
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * The token of the [ParsingSentence].
 *
 * @property id the id of the token, unique within its sentence
 * @property form the form
 * @property morphologies the list of possible morphologies of the token
 * @property pos the list of part-of-speech tags associated to the token (more for composite tokens, can be null)
 * @param position the position of the token in the text (null if it is a trace)
 */
data class ParsingToken(
  override val id: Int,
  override val form: String,
  override val morphologies: List<Morphology>,
  override val pos: List<POSTag>? = null, // TODO: find a better solution
  private val position: Position?
) : MorphoToken, FormToken, TokenIdentificable
