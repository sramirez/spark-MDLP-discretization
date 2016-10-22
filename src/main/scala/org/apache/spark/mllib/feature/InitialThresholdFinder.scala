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

import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import InitialThresholdFinder.isBoundary

object InitialThresholdFinder {
  /**
    * @return true if f1 and f2 define a boundary.
    *   It is a boundary if there is more than one class label present when the two are combined.
    */
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
    (f1, f2).zipped.map(_ + _).count(_ != 0) > 1
  }
}

/**
  * Find the initial thresholds. Look at the unique points and find ranges where the distribution
  * of labels is identical and split only where they are not.
  */
class InitialThresholdFinder() extends Serializable{

  /**
    * Computes the initial candidate cut points by feature.
    *
    * @param points RDD with distinct points by feature ((feature, point), class values).
    * @return RDD of candidate points.
    */
  def findInitialThresholds(points: RDD[((Int, Float), Array[Long])],
                                    nFeatures: Int, nLabels: Int, maxByPart: Int) = {

    val featureInfo = createFeatureInfoMap(points, maxByPart)
    val totalPartitions = featureInfo.last._5 + featureInfo.last._6

    // Get the first element cuts and their order index by partition for the boundary points evaluation
    val pointsWithIndex = points.zipWithIndex().map(  v => ((v._1._1._1, v._1._1._2, v._2), v._1._2))

    /** This custom partitioner will partition by feature and subdivide features into smaller partitions if large */
    class FeaturePartitioner[V]()
      extends Partitioner {

      def getPartition(key: Any): Int = {
        val (featureIdx, cut, sortIdx) = key.asInstanceOf[(Int, Float, Long)]
        val (_, _, sumValuesBefore, partitionSize, _, sumPreviousNumParts) = featureInfo(featureIdx)
        val partKey = sumPreviousNumParts + (Math.max(0, sortIdx - sumValuesBefore - 1) / partitionSize).toInt
        partKey
      }

      override def numPartitions: Int = totalPartitions
    }

    // partition by feature (possibly sub-partitioned features) instead of default partitioning strategy
    val partitionedPoints = pointsWithIndex.partitionBy(new FeaturePartitioner())

    partitionedPoints.mapPartitionsWithIndex({ (index, it) =>
      if (it.hasNext) {
        var ((lastFeatureIdx, lastX, _), lastFreqs) = it.next()
        //println("the first value of part " + index + " is " + lastX)
        var result = Seq.empty[((Int, Float), Array[Long])]
        var accumFreqs = lastFreqs

        for (((fIdx, x, _), freqs) <- it) {
          if (isBoundary(freqs, lastFreqs)) {
            // new boundary point: midpoint between this point and the previous one
            result = ((lastFeatureIdx, midpoint(x, lastX)), accumFreqs.clone) +: result
            accumFreqs = Array.fill(nLabels)(0L)
          }

          lastX = x
          lastFeatureIdx = fIdx
          lastFreqs = freqs
          accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
        }

        // The last X is either on a feature or a partition boundary
        result = ((lastFeatureIdx, lastX), accumFreqs.clone) +: result
        result.reverse.toIterator
      } else {
        Iterator.empty
      }
    })
  }

  /**
    * @param points all unique points
    * @param maxByPart maximum number of values in a partition
    * @return a list of info for each partition. The values in the info tuple are:
    *  (featureIdx, numUniqueValues, sumValsBeforeFirst, partitionSize, numPartitionsForFeature, sumPreviousPartitions)
    */
  def createFeatureInfoMap(points: RDD[((Int, Float), Array[Long])],
                           maxByPart: Int): List[(Int, Long, Long, Int, Int, Int)] = {
    // First find the number of points in each partition
    val countsByFeatureIdx = points.map(_._1._1).countByValue().toList.sortBy(_._1)

    var lastCount: Long = 0
    var sum: Long = 0
    var sumPreviousNumParts: Int = 0

    countsByFeatureIdx.map(x => {
      val partSize = Math.ceil(x._2 / Math.ceil(x._2 / maxByPart.toFloat)).toInt
      val numParts = Math.ceil(x._2 / partSize.toFloat).toInt
      val info = (x._1, x._2, sum + lastCount, partSize, numParts, sumPreviousNumParts)
      sum += lastCount
      sumPreviousNumParts += numParts
      lastCount = x._2
      info
    })
  }

  // If one of the unique values is NaN, use the other one, otherwise take the midpoint.
  def midpoint(x1: Float, x2: Float): Float = {
    if (x1.isNaN) x2
    else if (x2.isNaN) x1
    else (x1 + x2) / 2.0F
  }

}
