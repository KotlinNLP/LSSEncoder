/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder.decoder

import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.simplednn.simplemath.cosineSimilarity

/**
 * The [HeadsDecoder] based on the cosine similarity function.
 */
@Suppress("UNUSED")
class CosineDecoder : HeadsDecoder {

  companion object {

    /**
     * Half PI.
     */
    private const val HALF_PI = Math.PI / 2
  }

  /**
   * The private map of scored arcs.
   * Scores are mapped by dependents to governors ids (the root is intended to have id = -1).
   */
  private val similarityMatrix = mutableMapOf<Int, MutableMap<Int, Double>>()

  /**
   * The latent syntactic structure that contains the context-vectors and latent-heads used in the calculation of
   * the similarity.
   *
   * The root vector must be normalized each time because it is being trained.
   */
  private lateinit var lssNorm: LatentSyntacticStructure<*, *>

  /**
   * Decode all the possible heads of each encoded token, assigning them a score.
   * The decoding is done calculating the similarity scores among the context vectors, the latent heads and the root
   * vector.
   *
   * @param lss the latent syntactic structure of a sentence
   *
   * @return the scored heads
   */
  override fun decode(lss: LatentSyntacticStructure<*, *>): ScoredArcs {

    this.lssNorm = lss.copy(
      contextVectors = lss.contextVectors.map { it.normalize2() },
      latentHeads = lss.latentHeads.map { it.normalize2() },
      virtualRoot = lss.virtualRoot.normalize2())

    lss.sentence.tokens.forEach {

      this.similarityMatrix[it.id] = mutableMapOf()

      this.setHeadsScores(it)
      this.setRootScore(it)
      this.normalizeToDistribution(it)
    }

    return ScoredArcs(scores = this.similarityMatrix)
  }

  /**
   * Set the heads scores of the given [dependent] in the [similarityMatrix] map.
   *
   * @param dependent the dependent token
   */
  private fun <T : TokenIdentificable>setHeadsScores(dependent: T) {

    val scores: MutableMap<Int, Double> = this.similarityMatrix.getValue(dependent.id)

    this.lssNorm.sentence.tokens
      .asSequence()
      .filter { it.id != dependent.id }
      .associateTo(scores) {
        it.id to cosineSimilarity(
          a = this.lssNorm.getContextVectorById(it.id),
          b = this.lssNorm.getLatentHeadById(dependent.id))
      }
  }

  /**
   * Set the root score of the given [dependent] in the [similarityMatrix] map.
   *
   * @param dependent the dependent token
   */
  private fun <T : TokenIdentificable>setRootScore(dependent: T) {

    this.similarityMatrix.getValue(dependent.id)[ScoredArcs.rootId] = 0.0 // default root score

    if (dependent is FormToken && !dependent.isPunctuation) { // the root shouldn't be a punctuation token

      this.similarityMatrix.getValue(dependent.id)[ScoredArcs.rootId] = cosineSimilarity(
        a = this.lssNorm.getLatentHeadById(dependent.id),
        b = this.lssNorm.virtualRoot)
    }
  }

  /**
   * Normalize the scores of the given [dependent].
   * Scores are transformed into a linear scale with the arc cosine function and normalized into a probability
   * distribution.
   *
   * @param dependent the dependent token
   */
  private fun <T : TokenIdentificable>normalizeToDistribution(dependent: T) {

    val scores: MutableMap<Int, Double> = this.similarityMatrix.getValue(dependent.id)

    scores.forEach { scores.compute(it.key) { _, _ -> HALF_PI - Math.acos(it.value) } }

    val normSum: Double = scores.values.sum()

    scores.forEach { scores.compute(it.key) { _, _ -> it.value / normSum } }
  }
}
