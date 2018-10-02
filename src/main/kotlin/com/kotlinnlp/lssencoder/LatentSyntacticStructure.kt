/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder

import com.kotlinnlp.lssencoder.language.ParsingSentence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The latent syntactic structure encoded by the tokens encoder, the context encoder and the heads encoder.
 *
 * @property sentence the sentence containing the tokens
 * @property contextVectors the tokens encoded by the tokens encoder
 * @property contextVectors the context vectors encoded by the context encoder
 * @property latentHeads the latent heads encoded by the heads encoder
 * @property virtualRoot the vector that represents the root token of a sentence
 */
data class LatentSyntacticStructure(
  val sentence: ParsingSentence,
  val tokensEncodings: List<DenseNDArray>,
  val contextVectors: List<DenseNDArray>,
  val latentHeads: List<DenseNDArray>,
  val virtualRoot: DenseNDArray
) {

  /**
   * The length of the sentence.
   */
  val size: Int = this.sentence.tokens.size

  /**
   * The latent syntactic encodings of the [sentence] tokens, obtained as concatenation of the context vectors with the
   * related latent head representation.
   */
  val latentSyntacticEncodings: List<DenseNDArray> by lazy {
    this.contextVectors.zip(this.latentHeads) { contextVector, latentHead -> contextVector.concatV(latentHead) }
  }

  /**
   * @param id the id of a token of the [sentence]
   *
   * @return the encoding of the given token
   */
  fun getTokenEncodingById(id: Int): DenseNDArray = this.tokensEncodings[this.sentence.getTokenIndex(id)]

  /**
   * @param id the id of a token of the [sentence]
   *
   * @return the context vector of the given token
   */
  fun getContextVectorById(id: Int): DenseNDArray = this.contextVectors[this.sentence.getTokenIndex(id)]

  /**
   * @param id the id of a token of the [sentence]
   *
   * @return the latent head of the given token
   */
  fun getLatentHeadById(id: Int): DenseNDArray = this.latentHeads[this.sentence.getTokenIndex(id)]

  /**
   * @param id the id of a token of the [sentence]
   *
   * @return the latent syntactic encoding of the given token
   */
  fun getLSEncodingById(id: Int): DenseNDArray = this.latentSyntacticEncodings[this.sentence.getTokenIndex(id)]
}
