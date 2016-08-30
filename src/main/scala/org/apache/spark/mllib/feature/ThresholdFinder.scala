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
import FeatureUtils._


/**
  * Base trait for threshold finders.
  */
trait ThresholdFinder extends Serializable with Logging {

  /**
    * @param bucketInfo info about the parent bucket
    * @param leftFreqs frequencies to the left
    * @param rightFreqs frequencies to the right
    * @return the MDLP criterion value, and the weighted entropy value
    */
  def calcCriterionValue(bucketInfo: BucketInfo,
                         leftFreqs: Seq[Long], rightFreqs: Seq[Long]): (Double, Double, Long, Long) = {
    val k1 = leftFreqs.count(_ != 0)
    val s1 = if (k1 > 0) leftFreqs.sum else 0
    val hs1 = entropy(leftFreqs, s1)
    val k2 = rightFreqs.count(_ != 0)
    val s2 = if (k2 > 0) rightFreqs.sum else 0
    val hs2 = entropy(rightFreqs, s2)
    val weightedHs = (s1 * hs1 + s2 * hs2) / bucketInfo.s
    val gain = bucketInfo.hs - weightedHs
    val delta = log2(math.pow(3, bucketInfo.k) - 2) - (bucketInfo.k * bucketInfo.hs - k1 * hs1 - k2 * hs2)
    (gain - (log2(bucketInfo.s - 1) + delta) / bucketInfo.s, weightedHs, s1, s2)
  }

}
