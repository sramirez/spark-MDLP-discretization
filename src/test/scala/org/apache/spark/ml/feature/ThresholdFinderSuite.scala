package org.apache.spark.ml.feature

import org.apache.spark.mllib.feature.{BucketInfo, ThresholdFinder}
import org.apache.spark.sql.SQLContext
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner


/**
  * Test entropy calculation.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class ThresholdFinderSuite extends FunSuite {


  test("Test calcCriterion with even split hence low criterion value (and high entropy)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(50L, 100L, 150L)
    val rightFreqs = Array(50L, 100L, 150L)

    assertResult((-0.030412853556075408, 1.4591479170272448, 300, 300)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }

  test("Test calcCriterion with even split (and some at split) hence low criterion value (and high entropy)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(40L, 100L, 140L)
    val rightFreqs = Array(50L, 90L, 150L)

    assertResult((0.05852316831964029,1.370380206618117,280,290)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }

  test("Test calcCriterion with uneven split hence high criterion value (and low entropy)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(100L, 10L, 250L)
    val rightFreqs = Array(0L, 190L, 50L)

    assertResult((0.5270800719912969, 0.9086741857687387, 360, 240)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }

  test("Test calcCriterion with uneven split hence very high criterion value (and very low entropy)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(100L, 200L, 0L)
    val rightFreqs = Array(0L, 0L, 300L)

    assertResult((0.9811176395006821, 0.45914791702724483, 300, 300)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }

  test("Test calcCriterion with all data on one side (hence low criterion value)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(0L, 0L, 0L)
    val rightFreqs = Array(100L, 200L, 300L)

    assertResult((-0.02311711397093918, 1.4591479170272448, 0, 600)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }

  test("Test calcCriterion with most data on one side (hence low criterion value)") {

    val bucketInfo = new BucketInfo(Array(100L, 200L, 300L))
    val leftFreqs = Array(0L, 10L, 0L)
    val rightFreqs = Array(100L, 190L, 300L)

    assertResult((0.003721577231942788,1.4323219723298557,10,590)) {
      ThresholdFinder.calcCriterionValue(bucketInfo, leftFreqs, rightFreqs)
    }
  }
}