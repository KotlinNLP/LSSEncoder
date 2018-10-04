/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.lssencoder

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNParameters
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer

/**
 * The optimizer of the [LSSModel].
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class LSSOptimizer(
  private val model: LSSModel<*, *>,
  updateMethod: UpdateMethod<*>
) : Optimizer<LSSParameters>(updateMethod) {

  /**
   * The optimizer of the tokens encoder.
   */
  private val tokensEncoderOptimizer: TokensEncoderOptimizer =
    this.model.tokensEncoderWrapperModel.model.buildOptimizer(updateMethod)

  /**
   * The optimizer of the heads encoder.
   */
  private val contextEncoderOptimizer: ParamsOptimizer<DeepBiRNNParameters> =
    ParamsOptimizer(params = this.model.contextEncoderModel.model, updateMethod = updateMethod)

  /**
   * The optimizer of the heads encoder.
   */
  private val headsEncoderOptimizer: ParamsOptimizer<BiRNNParameters> =
    ParamsOptimizer(params = this.model.headsEncoderBiRNN.model, updateMethod = updateMethod)

  /**
   * Update the parameters of the neural modules of the [model].
   */
  override fun update() {
    this.tokensEncoderOptimizer.update()
    this.contextEncoderOptimizer.update()
    this.headsEncoderOptimizer.update()
  }

  /**
   * Accumulate the given [paramsErrors].
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: LSSParameters, copy: Boolean) {
    this.tokensEncoderOptimizer.accumulate(paramsErrors = paramsErrors.tokensEncoderParams, copy = copy)
    this.contextEncoderOptimizer.accumulate(paramsErrors = paramsErrors.contextEncoderParams, copy = copy)
    this.headsEncoderOptimizer.accumulate(paramsErrors = paramsErrors.headsEncoderParams, copy = copy)
  }
}
