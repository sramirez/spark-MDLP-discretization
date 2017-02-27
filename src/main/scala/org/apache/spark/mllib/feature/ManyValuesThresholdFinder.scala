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
import scala.collection.mutable
import ThresholdFinder.calcCriterionValue


/**
  * Use this version when the feature to discretize has more values that will fit in a partition (see maxByPart param).
  * @param nLabels the number of class labels
  * @param maxBins Maximum number of points to select
  * @param minBinWeight don't generate bins with fewer than this many records.
  * @param elementsByPart Maximum number of elements to evaluate in each partition.
  * @param stoppingCriterion influences when to stop recursive splits
  */
class ManyValuesThresholdFinder(nLabels: Int, stoppingCriterion: Double,
                                maxBins: Int, minBinWeight: Long,
                                elementsByPart: Int)
  extends ThresholdFinder {

  /**
    * Evaluate boundary points and select the most relevant. This version is used when
    * the number of candidates exceeds the maximum size per partition (distributed version).
    *
    * @param candidates RDD of candidates points (point, class histogram).
    * @return Sequence of threshold values.
    */
  def findThresholds(candidates: RDD[(Float, Array[Long])]): Seq[Float] = {

    // Get the number of partitions according to the maximum size established by partition
    def partitions(x: Long) = math.ceil(x.toFloat / elementsByPart).toInt

    // Insert the extreme values in the stack (recursive iteration)
    val stack = new mutable.Queue[((Float, Float), Option[Float])]
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq(Float.NegativeInfinity) // # points = # bins - 1

    while (stack.nonEmpty && result.size < maxBins){
      val (bounds, lastThresh) = stack.dequeue
      // Filter the candidates between the last limits added to the stack
      val cands = candidates.filter({ case (th, _) => th > bounds._1 && th < bounds._2 })
      //val nCands = cands.count
      if (!cands.isEmpty()) {
        // There does not seem to be an advantage to doing this here, and it can give different results
        // Possibly because of SPARK-14393?
        //cands = cands.coalesce(partitions(nCands))
        // Selects one threshold among the candidates and returns two partitions to recurse
        evalThresholds(cands, lastThresh) match {
          case Some(th) =>
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => /* criteria not fulfilled, finish! */
        }
      }
    }

    result.sorted :+ Float.PositiveInfinity
  }

  /**
    * Compute entropy minimization for candidate points in a range,
    * and select the best one according to the MDLP criterion (RDD version).
    *
    * @param candidates RDD of candidate points (point, class histogram).
    * @param lastSelected Last selected threshold.
    * @return The minimum-entropy candidate.
    */
  private def evalThresholds(candidates: RDD[(Float, Array[Long])],
                             lastSelected : Option[Float]) = {

    val sc = candidates.sparkContext

    // Compute the accumulated frequencies by partition
    val totalsByPart = sc.runJob(candidates, { it =>
      val accum = Array.fill(nLabels)(0L)
      for ((_, freqs) <- it) {
        for (i <- 0 until nLabels) accum(i) += freqs(i)
      }
      accum
    }: (Iterator[(Float, Array[Long])]) => Array[Long])

    // Compute the total frequency for all partitions
    var totals = Array.fill(nLabels)(0L)
    for (t <- totalsByPart) totals = (totals, t).zipped.map(_ + _)
    val bcTotalsByPart = sc.broadcast(totalsByPart)
    val bcTotals = sc.broadcast(totals)

    val result = candidates.mapPartitionsWithIndex({ (slice, it) =>
      // Accumulate frequencies from the left to the current partition
      var leftTotal = Array.fill(nLabels)(0L)
      for (i <- 0 until slice) leftTotal = (leftTotal, bcTotalsByPart.value(i)).zipped.map(_ + _)
      var entropyFreqs = Seq.empty[(Float, Array[Long], Array[Long], Array[Long])]
      // ... and from the current partition to the rightmost partition
      for ((cand, freqs) <- it) {
        leftTotal = (leftTotal, freqs).zipped.map(_ + _)
        val rightTotal = (bcTotals.value, leftTotal).zipped.map(_ - _)
        entropyFreqs = (cand, freqs, leftTotal, rightTotal) +: entropyFreqs
      }
      entropyFreqs.iterator
    })

    val bucketInfo = new BucketInfo(totals)

    // select the best threshold according to MDLP
    val finalCandidates = result.flatMap({
      case (cand, _, leftFreqs, rightFreqs) =>
        val (criterionValue, weightedHs, leftSum, rightSum) = calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
        var criterion = criterionValue > stoppingCriterion && leftSum > minBinWeight && rightSum > minBinWeight
        lastSelected match {
          case None =>
          case Some(last) => criterion = criterion && (cand != last)
        }
        if (criterion) Seq((weightedHs, cand)) else Seq.empty[(Double, Float)]
    })
    // Select the candidate with the minimum weightedHs from among the list of accepted candidates.
    if (!finalCandidates.isEmpty()) Some(finalCandidates.min._2) else None
  }
}
