/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder

import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The [LSSModel] parameters.
 *
 * @property tokensEncoderParams the parameters of the tokens encoder
 * @property contextEncoderParams the parameters of the context encoder
 * @property headsEncoderParams the parameters of the heads encoder
 */
data class LSSParameters(
  val tokensEncoderParams: TokensEncoderParameters,
  val contextEncoderParams: DeepBiRNNParameters,
  val headsEncoderParams: BiRNNParameters
)
