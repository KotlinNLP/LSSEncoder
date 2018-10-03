/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatFeedforwardMerge
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.Serializable

/**
 * The model of the [LSSEncoder].
 *
 * @property language the language within the encoder works (default = unknown)
 * @property tokensEncoderWrapperModel the model of the TokensEncoder combined with its sentence converter
 * @property contextBiRNNConfig the configuration of the ContextEncoder BiRNN (if null the ContextEncoder is not used)
 * @property headsBiRNNConfig the configuration of the HeadsEncoder BiRNN
 */
class LSSModel<TokenType : TokenIdentificable, SentenceType : SentenceIdentificable<TokenType>>(
  val language: Language = Language.Unknown,
  val tokensEncoderWrapperModel: TokensEncoderWrapperModel<TokenType, SentenceType, *, *>,
  val contextBiRNNConfig: BiRNNConfig,
  val headsBiRNNConfig: BiRNNConfig
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [LSSModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [LSSModel]
     *
     * @return the [LSSModel] read from [inputStream] and decoded
     */
    fun <TokenType: TokenIdentificable, SentenceType: SentenceIdentificable<TokenType>>load(
      inputStream: InputStream
    ): LSSModel<TokenType, SentenceType> = Serializer.deserialize(inputStream)
  }

  /**
   * The model of the context encoder.
   */
  val contextEncoderModel = if (contextBiRNNConfig.numberOfLayers == 2)
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null),
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize * 2,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null))
  else
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokensEncoderWrapperModel.model.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null))

  /**
   * The size of the context vectors.
   */
  val contextVectorsSize: Int = this.contextEncoderModel.outputSize

  /**
   * The model of the heads encoder.
   */
  val headsEncoderBiRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = this.contextVectorsSize,
    dropout = 0.0,
    recurrentConnectionType = headsBiRNNConfig.connectionType,
    hiddenActivation = headsBiRNNConfig.hiddenActivation,
    hiddenSize = this.contextVectorsSize,
    outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = this.contextVectorsSize))

  /**
   * The embeddings vector that represents the root token of a sentence.
   */
  val rootEmbedding = Embedding(id = 0, array = UpdatableDenseArray(Shape(this.contextVectorsSize)))

  /**
   * Initialize the root embedding.
   */
  init {
    GlorotInitializer().initialize(this.rootEmbedding.array.values)
  }
}
