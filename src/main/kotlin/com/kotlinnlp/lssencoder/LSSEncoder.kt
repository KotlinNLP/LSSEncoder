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
import com.kotlinnlp.simplednn.simplemath.assignSum
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
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
   * @property size the number of errors in each list
   * @property tokensEncodings the errors of the tokens encodings (can be null)
   * @property contextVectors the errors of the context vectors (can be null)
   * @property latentHeads the errors of the latent heads (can be null)
   */
  data class OutputErrors(
    val size: Int,
    val tokensEncodings: List<DenseNDArray>? = null,
    val contextVectors: List<DenseNDArray>? = null,
    val latentHeads: List<DenseNDArray>? = null)

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
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings).map { it.copy() }
    val latentHeads: List<DenseNDArray> = this.headsEncoder.forward(contextVectors).map { it.copy() }

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

    val latentHeadsErrors: List<DenseNDArray> = outputErrors.latentHeads
      ?: List(size = outputErrors.size, init = { DenseNDArrayFactory.zeros(Shape(this.model.contextVectorsSize)) })
    val tokensEncodingsSize: Int = this.model.tokensEncoderWrapperModel.model.tokenEncodingSize
    val tokensEncodingsErrors: List<DenseNDArray> = outputErrors.tokensEncodings
      ?: List(size = outputErrors.size, init = { DenseNDArrayFactory.zeros(Shape(tokensEncodingsSize)) })
    val contextVectorsErrors: List<DenseNDArray> = outputErrors.contextVectors
      ?: List(size = outputErrors.size, init = { DenseNDArrayFactory.zeros(Shape(this.model.contextVectorsSize)) })

    this.headsEncoder.backward(latentHeadsErrors)

    contextVectorsErrors.assignSum(this.headsEncoder.getInputErrors(copy = false))
    this.contextEncoder.backward(contextVectorsErrors)

    tokensEncodingsErrors.assignSum(this.contextEncoder.getInputErrors(copy = false))
    this.tokensEncoderWrapper.backward(tokensEncodingsErrors)
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
