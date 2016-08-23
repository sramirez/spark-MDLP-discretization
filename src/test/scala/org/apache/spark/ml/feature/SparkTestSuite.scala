package org.apache.spark.ml.feature


import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import SparkTestSuite.SPARK_CTX

object SparkTestSuite {
  final val SPARK_CTX = createSparkContext()
  final val FILE_PREFIX = "src/test/resources/data/"


  def createSparkContext() = {
    // the [n] corresponds to the number of worker threads and should correspond ot the number of cores available.
    val conf = new SparkConf().setAppName("test-spark").setMaster("local[4]")
    // Changing the default parallelism gave slightly different results and did not do much for performance.
    //conf.set("spark.default.parallelism", "2")
    val sc = new SparkContext(conf)
    LogManager.getRootLogger.setLevel(Level.WARN)
    sc
  }
}

/**
  * This can be used to experiment with trying different ways of using spark.
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class SparkTestSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }


  /** do some simple processing to make sure spark is working and can be called from tests */
  test("Simple Spark test") {
    val sequence = Seq(1, 2, 3, 2, 1, 4)
    // assert that there are 4 numbers in the sequence that are <= 2
    assertResult(4)(SPARK_CTX.parallelize(sequence).filter(_ <= 2).map(_ + 1).count)
  }
}