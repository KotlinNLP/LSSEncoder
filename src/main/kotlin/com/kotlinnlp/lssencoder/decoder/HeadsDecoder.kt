/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder.decoder

import com.kotlinnlp.lssencoder.LatentSyntacticStructure

/**
 * The decoder of the heads encoded in a [LatentSyntacticStructure].
 */
interface HeadsDecoder {

  /**
   * Decode all the possible heads of each encoded token, assigning them a score.
   * The decoding is done calculating the similarity scores among the context vectors, the latent heads and the root
   * vector.
   *
   * @param lss the latent syntactic structure of a sentence
   *
   * @return the scored heads
   */
  fun decode(lss: LatentSyntacticStructure<*, *>): ScoredArcs
}
