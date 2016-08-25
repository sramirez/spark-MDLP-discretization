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

import FeatureUtils._

/**
  * @param totals frequency totals for unique values in the bucket
  */
class BucketInfo(totals: Array[Long]) extends Serializable {
  // number of elements in bucket
  val s = totals.sum

  // calculated entropy for the bucket
  val hs = entropy(totals.toSeq, s)

  // number of distinct classes in bucket
  val k = totals.count(_ != 0)
}
