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

import scala.collection.mutable
import ThresholdFinder.calcCriterionValue

/**
  * Use this version when the feature to discretize has relatively few unique values.
  * @param nLabels the number of class labels
  * @param maxBins Maximum number of points to select.
  * @param minBinWeight don't generate bins with fewer than this many records.
  * @param stoppingCriterion influences when to stop recursive splits
  */
class FewValuesThresholdFinder(nLabels: Int, stoppingCriterion: Double, maxBins: Int, minBinWeight: Long)
  extends ThresholdFinder  {

  /**
    * Evaluates boundary points and selects the most relevant candidates (sequential version).
    * Here, the evaluation is bounded by partition as the number of points is small enough.
    *
    * @param candidates RDD of candidates points (point, class histogram).
    * @return Sequence of threshold values.
    */
  def findThresholds(candidates: Array[(Float, Array[Long])]): Seq[Float] = {

    val stack = new mutable.Queue[((Float, Float), Option[Float])]
    // Insert first in the stack (recursive iteration)
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq(Float.NegativeInfinity) // # points = # bins - 1

    while (stack.nonEmpty && result.size < maxBins){
      val (bounds, lastThresh) = stack.dequeue
      // Filter the candidates between the last limits added to the stack
      val newCandidates = candidates.filter({ case (th, _) =>
        th > bounds._1 && th < bounds._2
      })
      if (newCandidates.length > 0) {
        //println("evaluating newCandidates = " + newCandidates.map(x=>x._1 + ": " + x._2.mkString(", ")).mkString(";   ") + "   lastThresh= " + lastThresh)
        evalThresholds(newCandidates, lastThresh, nLabels) match {
          case Some(th) =>
            //println("added "+ th + " to " + result)
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => /* criteria not fulfilled, finish */
        }
      }
    }
    result.sorted :+ Float.PositiveInfinity
  }

  /**
    * Compute entropy minimization for candidate points in a range,
    * and select the best one according to MDLP criterion (sequential version).
    *
    * @param candidates Array of candidate points (point, class histogram).
    * @param lastSelected last selected threshold.
    * @param nLabels Number of classes.
    * @return The minimum-entropy cut point.
    *
    */
  private def evalThresholds(candidates: Array[(Float, Array[Long])],
                              lastSelected : Option[Float],
                              nLabels: Int): Option[Float] = {

    // Calculate the total frequencies by label
    val totals = candidates.map(_._2).reduce((freq1, freq2) => (freq1, freq2).zipped.map(_ + _))

    // Compute the accumulated frequencies (both left and right) by label
    var leftAccum = Array.fill(nLabels)(0L)
    var entropyFreqs = Seq.empty[(Float, Array[Long], Array[Long], Array[Long])]
    for (i <- candidates.indices) {
      val (cand, freq) = candidates(i)
      leftAccum = (leftAccum, freq).zipped.map(_ + _)
      val rightTotal = (totals, leftAccum).zipped.map(_ - _)
      entropyFreqs = (cand, freq, leftAccum, rightTotal) +: entropyFreqs
    }
    //println(entropyFreqs.map(x => " cand=" + x._1 + "   f="+ x._2.mkString(", ") + "   left="+ x._3.mkString(", ") + "   right="+ x._4.mkString(", ") ).mkString("\n"))

    val bucketInfo = new BucketInfo(totals)

    // select best threshold according to the criteria
    val finalCandidates = entropyFreqs.flatMap({
      case (cand, _, leftFreqs, rightFreqs) =>
        val (criterionValue, weightedHs, leftSum, rightSum) = calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
        var criterion = criterionValue > stoppingCriterion && leftSum > minBinWeight && rightSum > minBinWeight

        lastSelected match {
          case None =>
          case Some(last) => criterion = criterion && (cand != last)
        }

        if (criterion) Seq((weightedHs, cand)) else Seq.empty[(Double, Float)]
    })
    // Select among the list of accepted candidate, that with the minimum weightedHs
    if (finalCandidates.nonEmpty) Some(finalCandidates.min._2) else None
  }

}
