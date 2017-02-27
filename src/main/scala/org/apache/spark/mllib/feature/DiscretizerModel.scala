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

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._

/**
 * Generic discretizer model that transforms data given a list of thresholds by feature.
 * 
 * @param thresholds Thresholds defined for each feature (must be sorted).
 *  
 * Note: checking the second sorting condition can be much more time-consuming. 
 * We omit this condition.
 */
class DiscretizerModel(val thresholds: Array[Array[Float]]) extends VectorTransformer {
  
  /**
   * Discretizes values in a given dataset using thresholds.
   *
   * @param data A single continuous-valued vector.
   * @return A resulting vector with its values discretized (from 1 to n).
   */
  override def transform(data: Vector) = {
    data match {
      case v: SparseVector =>
        val newValues = for (i <- v.indices.indices)
          yield assignDiscreteValue(v.values(i).toFloat, thresholds(v.indices(i))).toDouble
        
        // the `index` array inside sparse vector object will not be changed,
        // so we can re-use it to save memory.
        Vectors.sparse(v.size, v.indices, newValues.toArray)
        
        case v: DenseVector =>
          val newValues = for (i <- v.values.indices)
            yield assignDiscreteValue(v(i).toFloat, thresholds(i)).toDouble         
          Vectors.dense(newValues.toArray)
    }    
  }
  
  /**
   * Discretizes values in a given dataset using thresholds.
   *
   * @param data RDD with continuous-valued vectors.
   * @return RDD with discretized data (from 1 to n).
   */
  def transformRDD(data: RDD[Vector]) = {
    val bc_thresholds = data.context.broadcast(thresholds)    
    data.map {
      case v: SparseVector =>
        val newValues = for (i <- v.indices.indices)
          yield assignDiscreteValue(v.values(i).toFloat, bc_thresholds.value(v.indices(i))).toDouble
        
        // the `index` array inside sparse vector object will not be changed,
        // so we can re-use it to save memory.
        Vectors.sparse(v.size, v.indices, newValues.toArray)
        
        case v: DenseVector =>
          val newValues = for (i <- v.values.indices)
            yield assignDiscreteValue(v(i).toFloat, bc_thresholds.value(i)).toDouble         
          Vectors.dense(newValues.toArray)
    }
  }

  private def binarySearch[A <% Ordered[A]](a: Array[A], v: A) = {
    def recurse(low: Int, high: Int): Int = (low + high) / 2 match {
      case _ if high < low => high + 1
      case midHigher if a(midHigher) > v => recurse(low, midHigher - 1)
      case midLower if a(midLower) < v => recurse(midLower + 1, high)
      case mid => mid
    }
    recurse(0, a.length - 1)
  }

  /**
   * Discretizes a value with a set of intervals.
   *
   * @param value Value to be discretized.
   * @param thresholds Thresholds used to assign a discrete value
   * 
   * Note: The first threshold mut be -Infinity and the
    * last threshold must be always Positive Infinity.
   */
  private def assignDiscreteValue(value: Float, thresholds: Array[Float]) = {
    if (thresholds.length > 0) {
      binarySearch(thresholds, value).toFloat
    } else {
      value
    }
  }

}
