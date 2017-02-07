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

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg._

import scala.collection.mutable

/**
 * Feature utils for selector methods.
 */
@Experimental
object FeatureUtils {


  private val LOG2 = math.log(2)

  /** @return log base 2 of x */
  val log2 = { x: Double => math.log(x) / LOG2 }


  /**
    * Entropy is a measure of disorder. The higher the value, the closer to a purely random distribution.
    * The MDLP algorithm tries to find splits that will minimize entropy.
    * @param frequencies sequence of integer frequencies.
    * @param n the sum of all the frequencies in the list.
    * @return the total entropy
    */
  def entropy(frequencies: Seq[Long], n: Long): Double = {
    -frequencies.aggregate(0.0)(
      { case (h, q) => h + (if (q == 0) 0 else {
          val qn = q.toDouble / n
          qn * log2(qn)
        })
      },
      { case (h1, h2) => h1 + h2 }
    )
  }

}
