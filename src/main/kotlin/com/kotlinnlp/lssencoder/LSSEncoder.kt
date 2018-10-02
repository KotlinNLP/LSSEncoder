/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder

import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapper
import java.lang.RuntimeException

/**
 * The main encoder that builds the latent syntactic structure.
 *
 * @property model the encoder model
 * @property useDropout whether to apply the dropout during the [forward]
 * @property id an identification number useful to track a specific encoder
 */
class LSSEncoder<TokenType : TokenIdentificable, SentenceType : SentenceIdentificable<TokenType>>(
  val model: LSSModel<TokenType, SentenceType>,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  SentenceType, // InputType
  LatentSyntacticStructure<TokenType, SentenceType>, // OutputType
  LSSEncoder.OutputErrors, // ErrorsType
  NeuralProcessor.NoInputErrors, // InputErrorsType
  LSSParameters // ParamsType
  > {

  /**
   * The output errors of the LSS encoder.
   *
   * @property contextVectors the errors of the context vectors
   * @property latentHeads the errors of the latent heads
   */
  data class OutputErrors(val contextVectors: List<DenseNDArray>, val latentHeads: List<DenseNDArray>)

  /**
   * Property not used because the input is a sentence.
   */
  override val propagateToInput: Boolean = false

  /**
   * The tokens encoder wrapped with a sentence converter from the input sentence.
   */
  private val tokensEncoderWrapper: TokensEncoderWrapper<TokenType, SentenceType, *, *> =
    this.model.tokensEncoderWrapperModel.buildWrapper(useDropout = true)

  /**
   * The encoder of tokens encodings sentential context.
   */
  private val contextEncoder = DeepBiRNNEncoder<DenseNDArray>(
    network = this.model.contextEncoderModel,
    propagateToInput = true,
    useDropout = this.useDropout)

  /**
   * The encoder that generated the latent heads representation.
   */
  private val headsEncoder = BiRNNEncoder<DenseNDArray>(
    network = this.model.headsEncoderBiRNN,
    propagateToInput = true,
    useDropout = this.useDropout)

  /**
   * @param input the sentence to encode
   *
   * @return the latent syntactic structure
   */
  override fun forward(input: SentenceType): LatentSyntacticStructure<TokenType, SentenceType> {

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoderWrapper.forward(input)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings)
    val latentHeads: List<DenseNDArray> = this.headsEncoder.forward(contextVectors)

    return LatentSyntacticStructure(
      sentence = input,
      tokensEncodings = tokensEncodings,
      contextVectors = contextVectors,
      latentHeads = latentHeads,
      virtualRoot = this.model.rootEmbedding.array.values)
  }

  /**
   * Backward of the neural modules.
   */
  override fun backward(outputErrors: OutputErrors) {

    this.headsEncoder.backward(outputErrors.latentHeads)

    val contextErrors: List<DenseNDArray> =
      outputErrors.contextVectors.zip(this.headsEncoder.getInputErrors(copy = false)) { c, h -> c.sum(h) }

    this.contextEncoder.backward(contextErrors)

    this.tokensEncoderWrapper.backward(this.contextEncoder.getInputErrors(copy = false))
  }

  /**
   * This method should not be used because the input is a sentence.
   */
  override fun getInputErrors(copy: Boolean): NeuralProcessor.NoInputErrors {
    throw RuntimeException("The input errors of the LSS Encoder cannot be obtained because the input is a sentence.")
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean) = LSSParameters(
    contextEncoderParams = this.contextEncoder.getParamsErrors(copy = copy),
    headsEncoderParams = this.headsEncoder.getParamsErrors(copy = copy),
    tokensEncoderParams = this.tokensEncoderWrapper.getParamsErrors(copy = copy)
  )
}
