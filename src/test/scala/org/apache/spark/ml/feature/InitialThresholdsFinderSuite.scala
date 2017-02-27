package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.TestHelper._
import org.apache.spark.mllib.feature.{FewValuesThresholdFinder, InitialThresholdsFinder}
import org.apache.spark.sql.SQLContext
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}


/**
  * Test the finding of thresholds when not too many unique values.
  * This test suite was created in order to track down a source of non-determinism in this algorithm.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class InitialThresholdsFinderSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = _
  val finder = new InitialThresholdsFinder()

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  test("Find initial thresholds for a single feature") {

    val feature: Array[((Int, Float), Array[Long])] = Array(
      ((0, 4.0f), Array(1L, 2L, 3L)),
      ((0, 4.5f), Array(3L, 20L, 12L)),
      ((0, 4.6f), Array(8L, 18L, 2L)),
      ((0, 5.0f), Array(5L, 4L, 20L))
    )
    val points = sqlContext.sparkContext.parallelize(feature)
    val result = finder.findInitialThresholds(points, 1, nLabels = 3, maxByPart = 100)


    assertResult("4.25, 4.55, 4.8, 5.0") {
      result.collect().map(_._1._2).mkString(", ")
    }
  }


  test("Find initial thresholds for four features") {

    val feature: Array[((Int, Float), Array[Long])] = Array(
      ((0, 100.0f), Array(1L, 2L, 3L)),
      ((0, 150.0f), Array(3L, 20L, 12L)),
      ((0, 300.6f), Array(8L, 18L, 2L)),
      ((0, 400.0f), Array(5L, 4L, 20L)),

      ((1, 4.0f), Array(0L, 0L, 3L)),
      ((1, 4.5f), Array(0L, 0L, 4L)),
      ((1, 4.6f), Array(8L, 18L, 0L)),
      ((1, 5.0f), Array(5L, 4L, 20L)),

      ((2, 4.0f), Array(1L, 2L, 3L)),
      ((2, 4.5f), Array(0L, 20L, 0L)),
      ((2, 4.6f), Array(0L, 8L, 0L)),
      ((2, 5.0f), Array(5L, 0L, 20L)),

      ((3, 4.0f), Array(1L, 2L, 3L)),
      ((3, 4.5f), Array(0L, 8L, 0L)),
      ((3, 4.6f), Array(0L, 18L, 0L)),
      ((3, 5.0f), Array(0L, 28L, 0L))
    )
    val points = sqlContext.sparkContext.parallelize(feature)
    val result = finder.findInitialThresholds(points, nLabels = 3, maxByPart = 100, nFeatures = 4)

    assertResult("(0,125.0), (0,225.3), (0,350.3), (0,400.0), (1,4.55), (1,4.8), (1,5.0), (2,4.25), (2,4.8), (2,5.0), (3,4.25), (3,5.0)") {
      result.collect().map(_._1).mkString(", ")
    }
  }

  // In this case the features have more values that fit in a partition
  test("Find initial thresholds when more values than maxByPart") {

    val result = finder.findInitialThresholds(createPointsFor2Features, nLabels = 3, maxByPart = 5, nFeatures = 2)

    assertResult("(0,125.0), (0,225.0), (0,450.0), (0,700.0), (0,800.0), (0,1025.0), (0,1150.0), (0,1200.0), " +
      "(1,3.75), (1,4.55), (1,4.8), (1,5.5), (1,6.0), (1,8.05), (1,8.8), (1,9.5), (1,10.0), (1,12.05), (1,12.6)") {
      result.collect().map(_._1).mkString(", ")
    }
  }

  test("test createFeatureInfList for 2 features") {

    val points = createPointsFor2Features

    // the tuple in result list is
    //(featureIdx, numUniqueValues, sumValsBeforeFirst, partitionSize, numPartitionsForFeature, sumPreviousPartitions)
    assertResult("(0,8,0,8,1,0), (1,11,8,11,1,1)") {
      finder.createFeatureInfoList(points, 100, nFeatures = 2).mkString(", ")
    }

    assertResult("(0,8,0,8,1,0), (1,11,8,6,2,1)") {
      finder.createFeatureInfoList(points, 10, nFeatures = 2).mkString(", ")
    }

    assertResult("(0,8,0,4,2,0), (1,11,8,6,2,2)") {
      finder.createFeatureInfoList(points, 7, nFeatures = 2).mkString(", ")
    }

    assertResult("(0,8,0,4,2,0), (1,11,8,4,3,2)") {
      finder.createFeatureInfoList(points, 5, nFeatures = 2).mkString(", ")
    }

    assertResult("(0,8,0,2,4,0), (1,11,8,2,6,4)") {
      finder.createFeatureInfoList(points, 2, nFeatures = 2).mkString(", ")
    }
  }

  private def createPointsFor2Features = {
    val features: Array[((Int, Float), Array[Long])] = Array(
      ((0, 100.0f), Array(1L, 2L, 3L)),
      ((0, 150.0f), Array(3L, 20L, 12L)),
      ((0, 300.0f), Array(3L, 20L, 12L)),
      ((0, 600.0f), Array(5L, 4L, 20L)),
      ((0, 800.0f), Array(1L, 2L, 3L)),
      ((0, 950.0f), Array(1L, 2L, 3L)),
      ((0, 1100.0f), Array(8L, 18L, 2L)),
      ((0, 1200.0f), Array(5L, 4L, 20L)),

      ((1, 3.0f), Array(1L, 2L, 3L)),
      ((1, 4.5f), Array(3L, 20L, 12L)),
      ((1, 4.6f), Array(8L, 18L, 2L)),
      ((1, 5.0f), Array(5L, 4L, 20L)),
      ((1, 6.0f), Array(1L, 2L, 3L)),
      ((1, 7.5f), Array(3L, 20L, 12L)),
      ((1, 8.6f), Array(8L, 18L, 2L)),
      ((1, 9.0f), Array(5L, 4L, 20L)),
      ((1, 10.0f), Array(1L, 2L, 3L)),
      ((1, 11.5f), Array(3L, 20L, 12L)),
      ((1, 12.6f), Array(8L, 18L, 2L))
    )
    sqlContext.sparkContext.parallelize(features)
  }
}