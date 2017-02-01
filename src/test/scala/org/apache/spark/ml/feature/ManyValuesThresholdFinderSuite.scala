package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.TestHelper._
import org.apache.spark.mllib.feature.ManyValuesThresholdFinder
import org.apache.spark.sql.SQLContext
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}


/**
  * Test the finding of thresholds when many unique values.
  * This test suite was created in order to track down a source of non-determinism in this algorithm.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class ManyValuesThresholdFinderSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = _

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }


  test("Run many values finder on nLabels = 3 feature len = 4") {

    val finder = new ManyValuesThresholdFinder(nLabels = 3, stoppingCriterion = 0,
      maxBins = 100, minBinWeight = 1)

    val feature = SPARK_CTX.parallelize(List(
      (4.0f, Array(1L, 2L, 3L)),
      (5.0f, Array(5L, 4L, 20L)) ,
      (4.5f, Array(3L, 20L, 12L)),
      (4.6f, Array(8L, 18L, 2L))
    ))
    val result = finder.findThresholds(feature)

    assertResult("-Infinity, 5.0, Infinity") {
      result.mkString(", ")
    }
  }

  // val data = SPARK_CTX.textFile(FILE_PREFIX + "titanic2.data")

  test("Run many values finder on nLabels = 6 feature len = 35662") {

    val threshs = SPARK_CTX.textFile(FILE_PREFIX + "thresholds.data")
    val feature = threshs.map(line => line.split(",").map(elem => elem.trim))
      .map(x => (x(0).toFloat, Array(x(1).toLong, x(2).toLong, x(3).toLong, x(4).toLong, x(5).toLong, x(6).toLong)))


    val finder = new ManyValuesThresholdFinder(nLabels = 6, stoppingCriterion = 0,
      maxBins = 100, minBinWeight = 1)

    val result = finder.findThresholds(feature)

    assertResult(35662) {feature.count()}
    assertResult("-Infinity, -73.94241, -73.93588, -73.93573, -73.93254, -73.92902, 40.86836, 40.868473, 40.868515, 40.86856, 40.8686, 40.868683, 40.868702, 40.86873, 40.868744, 40.86876, 40.86877, 40.86879, 40.868797, 40.868843, 40.868862, 40.86892, 40.868965, 40.868988, 40.869007, 40.86902, 40.869026, 40.86903, 40.86906, 40.869125, 40.86914, 40.869194, 40.869225, 40.869373, 40.869404, 40.86954, 40.869568, 40.86996, 40.869972, 40.869976, 40.869995, 40.87004, 40.870064, 40.87018, 40.870213, 40.870377, 40.87039, 40.87056, 40.870586, 40.871216, 40.87123, 40.871254, 40.87127, 40.871773, 40.871796, 40.87188, 40.8719, 40.872375, 40.872383, 40.872627, 40.872658, 40.872726, 40.872734, 40.874496, 40.874523, 40.874844, 40.874855, 40.874893, 40.875046, 40.875095, 40.87513, 40.87538, 40.875397, 40.876053, 40.876095, 40.87618, 40.876198, 40.877007, 40.877033, 40.87749, 40.87751, 227028.5, 229116.5, 230945.5, 232787.0, 234977.0, 237980.0, 237990.5, 238223.5, 239589.5, 239697.5, 239725.0, 240252.0, 241159.0, 241167.5, 241386.0, 253270.0, 253327.0, 255966.0, 258988.5, Infinity") {
      result.mkString(", ")
    }
  }

}