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

import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.rdd._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import MDLPDiscretizer._
import org.apache.spark.broadcast.Broadcast

import scala.collection.Map


/**
 * Entropy minimization discretizer based on Minimum Description Length Principle (MDLP)
 * proposed by Fayyad and Irani in 1993 [1].
 * 
 * [1] Fayyad, U., & Irani, K. (1993). 
 * "Multi-interval discretization of continuous-valued attributes for classification learning."
 *
 * @param data RDD of LabeledPoint
 * @param stoppingCriterion (optional) used to determine when to stop recursive splitting
 */
class MDLPDiscretizer private (val data: RDD[LabeledPoint],
            stoppingCriterion: Double = DEFAULT_STOPPING_CRITERION) extends Serializable with Logging {

  private val labels2Int = data.map(_.label).distinct.collect.zipWithIndex.toMap
  private val nLabels = labels2Int.size

  /**
   * Computes the initial candidate points by feature.
   * 
   * @param points RDD with distinct points by feature ((feature, point), class values).
   * @param firstElements First elements in partitions 
   * @return RDD of candidate points.
   * 
   */
  private def initialThresholds(
      points: RDD[((Int, Float), Array[Long])], 
      firstElements: Array[Option[(Int, Float)]]) = {
    
    val numPartitions = points.partitions.length
    val bcFirsts = points.context.broadcast(firstElements)      

    points.mapPartitionsWithIndex({ (index, it) =>      
      if (it.hasNext) {
        var ((lastK, lastX), lastFreqs) = it.next()
        var result = Seq.empty[((Int, Float), Array[Long])]
        var accumFreqs = lastFreqs      
        
        for (((k, x), freqs) <- it) {           
          if (k != lastK) {
            // new attribute: add last point from the previous one
            result = ((lastK, lastX), accumFreqs.clone) +: result
            accumFreqs = Array.fill(nLabels)(0L)
          } else if (isBoundary(freqs, lastFreqs)) {
            // new boundary point: midpoint between this point and the previous one
            result = ((lastK, (x + lastX) / 2), accumFreqs.clone) +: result
            accumFreqs = Array.fill(nLabels)(0L)
          }
          
          lastK = k
          lastX = x
          lastFreqs = freqs
          accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
        }
       
        // Evaluate the last point in this partition with the first one in the next partition
        val lastPoint = if (index < (numPartitions - 1)) {
          bcFirsts.value(index + 1) match {
            case Some((k, x)) => if (k != lastK) lastX else (x + lastX) / 2
            case None => lastX // last point in the attribute
          }
        }else{
            lastX // last point in the dataset
        }                    
        (((lastK, lastPoint), accumFreqs.clone) +: result).reverse.toIterator
      } else {
        Iterator.empty
      }             
    })
  }

  /**
   * Run the entropy minimization discretizer on input data.
   * 
   * @param contFeat Indices to discretize (if not specified, the algorithm tries to figure it out).
   * @param elementsByPart Maximum number of elements to keep in each partition.
   * @param maxBins Maximum number of thresholds per feature.
   * @return A discretization model with the thresholds by feature.
   */
  def runAll(
      contFeat: Option[Seq[Int]], 
      elementsByPart: Int,
      maxBins: Int): DiscretizerModel = {
    
    if (data.getStorageLevel == StorageLevel.NONE)
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")

    if (!data.filter(_.label == null).isEmpty())
      logError("Some null values have been found in the labelColumn."
          + " This problem must be fixed before continuing with discretization.")

    // Basic info. about the dataset
    val sc = data.context
    val bLabels2Int = sc.broadcast(labels2Int)
    val classDistrib = data.map(d => bLabels2Int.value(d.label)).countByValue()
    val bclassDistrib = sc.broadcast(classDistrib)
    val (dense, nFeatures) = data.first.features match {
      case v: DenseVector => 
        (true, v.size)
      case v: SparseVector =>
        (false, v.size)
    }
            
    val continuousVars = processContinuousAttributes(contFeat, nFeatures)
    logInfo("Number of continuous attributes: " + continuousVars.distinct.length)
    logInfo("Total number of attributes: " + nFeatures)      
    if (continuousVars.isEmpty) logWarning("Discretization aborted. " +
      "No continuous attributes in the dataset")
    
    // Generate pairs ((feature, point), class histogram)
    sc.broadcast(continuousVars)
    val featureValues =
        data.flatMap({
          case LabeledPoint(label, dv: DenseVector) =>
            val c = Array.fill[Long](nLabels)(0L)
            c(bLabels2Int.value(label)) = 1L
            for (i <- dv.values.indices) yield ((i, dv(i).toFloat), c)
          case LabeledPoint(label, sv: SparseVector) =>
            val c = Array.fill[Long](nLabels)(0L)
            c(bLabels2Int.value(label)) = 1L
            for (i <- sv.indices.indices) yield ((sv.indices(i), sv.values(i).toFloat), c)
        })
    
    // Group elements by feature and point (get distinct points)
    val nonZeros = featureValues.reduceByKey{ case (v1, v2) =>
      (v1, v2).zipped.map(_ + _)
    }

    val zeros = addZerosIfNeeded(nonZeros, bclassDistrib)
    val distinctValues = nonZeros.union(zeros)
    
    // Sort these values to perform the boundary points evaluation
    val sortedValues = distinctValues.sortByKey()
          
    // Get the first elements by partition for the boundary points evaluation
    val firstElements = sc.runJob(sortedValues, { case it =>
      if (it.hasNext) Some(it.next()._1) else None
    }: (Iterator[((Int, Float), Array[Long])]) => Option[(Int, Float)])
      
    // Filter those features selected by the user
    val arr = Array.fill(nFeatures) { false }
    continuousVars.foreach(arr(_) = true)
    val barr = sc.broadcast(arr)

    // Get only boundary points from the whole set of distinct values
    val initialCandidates = initialThresholds(sortedValues, firstElements)
      .map{case ((k, point), c) => (k, (point, c))}
      .filter({case (k, _) => barr.value(k)})
      .cache() // It will be iterated for "big" features

    val allThresholds: Array[(Int, Seq[Float])] = findAllThresholds(elementsByPart, maxBins, initialCandidates, sc)

    buildModelFromThresholds(nFeatures, continuousVars, allThresholds)
  }

  /**
    * Add zeros if dealing with sparse data
    *
    * @return rdd with 0's filled in
    */
  def addZerosIfNeeded(nonZeros: RDD[((Int, Float), Array[Long])],
                       bclassDistrib: Broadcast[Map[Int, Long]]): RDD[((Int, Float), Array[Long])] = {
    nonZeros
      .map { case ((k, p), v) => (k, v) }
      .reduceByKey { case (v1, v2) => (v1, v2).zipped.map(_ + _) }
      .map { case (k, v) =>
        val v2 = for (i <- v.indices) yield bclassDistrib.value(i) - v(i)
        ((k, 0.0F), v2.toArray)
      }.filter { case (_, v) => v.sum > 0 }
  }


  /**
    * Divide RDD into two categories according to the number of points by feature.
    * @return find threshold for both sorts of attributes - those with many values, and those with few.
    */
  def findAllThresholds(elementsByPart: Int, maxBins: Int,
                        initialCandidates: RDD[(Int, (Float, Array[Long]))],
                        sc: SparkContext): Array[(Int, Seq[Float])] = {
    val bigIndexes = initialCandidates
      .countByKey()
      .filter{case (_, c) => c > elementsByPart}
    val bBigIndexes = sc.broadcast(bigIndexes)

    val smallThresholds = findSmallThresholds(maxBins, initialCandidates, bBigIndexes)
    val bigThresholds = findBigThresholds(elementsByPart, maxBins, initialCandidates, bigIndexes)

    // Join all thresholds in a single structure
    val bigThRDD = sc.parallelize(bigThresholds.toSeq)
    val allThresholds = smallThresholds.union(bigThRDD).collect()
    allThresholds
  }

  /**
    * Feature with too many points must be processed iteratively (rare condition)
    *
    * @return the splits for featuers with more values than will fit in a partition.
    */
  def findBigThresholds(elementsByPart: Int, maxBins: Int,
                        initialCandidates: RDD[(Int, (Float, Array[Long]))],
                        bigIndexes: Map[Int, Long]): Map[Int, Seq[Float]] = {
    logInfo("Number of features that exceed the maximum size per partition: " +
      bigIndexes.size)

    var bigThresholds = Map.empty[Int, Seq[Float]]
    val bigThresholdsFinder = new ManyValuesThresholdFinder(nLabels, stoppingCriterion)
    for (k <- bigIndexes.keys) {
      val cands = initialCandidates.filter { case (k2, _) => k == k2 }.values.sortByKey()
      bigThresholds += ((k, bigThresholdsFinder.findThresholds(cands, maxBins, elementsByPart)))
    }
    bigThresholds
  }

  /**
    * The features with a small number of points can be processed in a parallel way
    *
    * @return the splits for features with few values
    */
  def findSmallThresholds(maxBins: Int,
                          initialCandidates: RDD[(Int, (Float, Array[Long]))],
                          bBigIndexes: Broadcast[Map[Int, Long]]): RDD[(Int, Seq[Float])] = {
    val smallThresholdsFinder = new FewValuesThresholdFinder(nLabels, stoppingCriterion)
    initialCandidates
      .filter { case (k, _) => !bBigIndexes.value.contains(k) }
      .groupByKey()
      .mapValues(_.toArray)
      .mapValues(points => smallThresholdsFinder.findThresholds(points.sortBy(_._1), maxBins))
  }

  def buildModelFromThresholds(nFeatures: Int, continuousVars: Array[Int], allThresholds: Array[(Int, Seq[Float])]): DiscretizerModel = {
    // Update the full list of features with the thresholds calculated
    val thresholds = Array.fill(nFeatures)(Array.empty[Float]) // Nominal values (empty)
    // Not processed continuous attributes
    continuousVars.foreach(f => thresholds(f) = Array(Float.PositiveInfinity))
    // Continuous attributes (> 0 cut point)
    allThresholds.foreach({ case (k, vth) =>
      thresholds(k) = if (nFeatures > 0) vth.toArray else Array(Float.PositiveInfinity)
    })
    logInfo("Number of features with thresholds computed: " + allThresholds.length)
    logDebug("thresholds = " + thresholds.map(_.mkString(", ")).mkString(";\n"))

    new DiscretizerModel(thresholds)
  }

}

/** Companion object for static members */
object MDLPDiscretizer {

  /** The original paper suggested 0 for the stopping criterion, but smaller values like -1e-3 yield more splits */
  private val DEFAULT_STOPPING_CRITERION = 0

  /** @return true if f1 and f2 define a boundary */
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
    (f1, f2).zipped.map(_ + _).count(_ != 0) > 1
  }

  /**
    * Get information about the attributes before performing discretization.
    *
    * @param contIndices Indexes to discretize (if not specified, they are calculated).
    * @param nFeatures Total number of input features.
    * @return Indexes of continuous features.
    */
  private def processContinuousAttributes(contIndices: Option[Seq[Int]],
                                          nFeatures: Int) = {
    contIndices match {
      case Some(s) =>
        // Attributes are in range 0..nFeatures
        val intersect = (0 until nFeatures).seq.intersect(s)
        require(intersect.size == s.size)
        s.toArray
      case None =>
        (0 until nFeatures).toArray
    }
  }

  /**
   * Train a entropy minimization discretizer given an RDD of LabeledPoints.
   * 
   * @param input RDD of LabeledPoint's.
   * @param continuousFeaturesIndexes Indexes of features to be discretized. 
   * If it is not provided, the algorithm selects those features with more than 
   * 256 (byte range) distinct values.
   * @param maxBins Maximum number of thresholds to select per feature.
   * @param maxByPart Maximum number of elements by partition.
   * @param stoppingCriterion the threshold used to determine when stop recursive splitting of buckets.
   * @return A DiscretizerModel with the subsequent thresholds.
   */
  def train(
      input: RDD[LabeledPoint],
      continuousFeaturesIndexes: Option[Seq[Int]] = None,
      maxBins: Int = 15,
      maxByPart: Int = 100000,
      stoppingCriterion: Double = DEFAULT_STOPPING_CRITERION) = {
    new MDLPDiscretizer(input, stoppingCriterion).runAll(continuousFeaturesIndexes, maxByPart, maxBins)
  }
}
