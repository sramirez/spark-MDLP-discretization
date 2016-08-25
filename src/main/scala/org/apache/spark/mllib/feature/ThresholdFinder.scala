/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import org.apache.spark.Logging

/**
  * Base trait for threshold finders.
  */
trait ThresholdFinder extends Serializable with Logging {

  private val LOG2 = math.log(2)

  /** @return log base 2 of x */
  val log2 = { x: Double => math.log(x) / LOG2 }

  /**
    * @param freqs sequence of integer frequencies.
    * @param n the sum of all the frequencies in the list.
    * @return the total entropy
    */
  def entropy(freqs: Seq[Long], n: Long) = {
    -freqs.aggregate(0.0)(
      { case (h, q) => h + (if (q == 0) 0 else (q.toDouble / n) * log2(q.toDouble / n))},
      { case (h1, h2) => h1 + h2 }
    )
  }

  def calcCriterionValue(s: Double, hs: Double, k: Long,
                         leftFreqs: Seq[Long], rightFreqs: Seq[Long]): (Double, Double) = {
    val k1 = leftFreqs.count(_ != 0)
    val s1 = if (k1 > 0) leftFreqs.sum else 0
    val hs1 = entropy(leftFreqs, s1)
    val k2 = rightFreqs.count(_ != 0)
    val s2 = if (k2 > 0) rightFreqs.sum else 0
    val hs2 = entropy(rightFreqs, s2)
    val weightedHs = (s1 * hs1 + s2 * hs2) / s
    val gain = hs - weightedHs
    val delta = log2(math.pow(3, k) - 2) - (k * hs - k1 * hs1 - k2 * hs2)
    (gain - (log2(s - 1) + delta) / s, weightedHs)
  }

}
