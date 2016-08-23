package org.apache.spark.ml

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import SparkTestSuite.{FILE_PREFIX, SPARK_CTX}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{DiscretizerModel, MDLPDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import MDLPDiscretizerSuite.CLEAN_SUFFIX

object MDLPDiscretizerSuite {
  val CLEAN_SUFFIX: String = "_CLEAN"
}
/**
  * This can be used to experiment with trying different ways of using spark.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class MDLPDiscretizerSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  /** Do entropy based binning of cars data from UC Irvine repository. */
  test("Run MDLPD on single mpg column in cars data (maxBins = 10") {

    val df = readCarsData()
    val model = getDiscretizerModel(df, Array("mpg"), "origin", 10)

    assertResult("16.1, 21.05, 30.95, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  /** Its an error if maxBins is less than 2 */
  test("Run MDLPD on single mpg column in cars data (maxBins = 2") {

    val df = readCarsData()
    val model = getDiscretizerModel(df, Array("mpg"), "origin", 2)

    assertResult("16.1, 21.05, 30.95, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  /**
    * Do entropy based binning of cars data for all the numeric columns using origin as target.
    * The algorithm may produce different splits for individual columns when several features are discretized
    * at once because of the interaction between features.
    */
  test("Run MDLPD on all columns in cars data (maxBins = 100, label=origin") {

    val df = readCarsData()
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "origin", 100)

    assertResult(
      """16.1, 21.05, 30.95, Infinity;
        |5.5, Infinity;
        |97.5, 169.5, Infinity;
        |78.5, 134.0, Infinity;
        |2379.5, 2959.5, Infinity;
        |13.5, 19.5, Infinity;
        |1980.5, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in cars data (maxBins = 100, label=brand") {

    val df = readCarsData()
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "brand", 100)

    assertResult(
      """21.05, Infinity;
        |5.5, Infinity;
        |120.5, 134.5, Infinity;
        |78.5, Infinity;
        |2550.5, Infinity;
        |Infinity;
        |Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  def getDiscretizerModel(df: DataFrame, inputCols: Array[String],
                          labelColumn: String, maxBins: Int = 100): DiscretizerModel = {
    val labelIndexer = new StringIndexer()
      .setInputCol(labelColumn)
      .setOutputCol(labelColumn + CLEAN_SUFFIX).fit(df)

    var processedDf = labelIndexer.transform(df)

    val featureAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
    processedDf = featureAssembler.transform(processedDf)

    val discretizer = new MDLPDiscretizer()
      .setMaxBins(maxBins)
      .setMaxByPart(10000)
      .setInputCol("features")  // this must be a feature vector
      .setLabelCol(labelColumn + CLEAN_SUFFIX)
      .setOutputCol("bucketFeatures")

    discretizer.fit(processedDf) //.transform(df)
  }

  /** @return the cars data as a dataframe */
  def readCarsData(): DataFrame = {
    val cars = SPARK_CTX.textFile(FILE_PREFIX + "cars.data")
    val nullable = true

    // mpg, cylinders, cubicinches, horsepower, weightlbs, time to sixty, year, brand, origin
    val schema = StructType(List(
      StructField("mpg", DoubleType, nullable),
      StructField("cylinders", IntegerType, nullable),
      StructField("cubicinches", IntegerType, nullable),
      StructField("horsepower", DoubleType, nullable),
      StructField("weightlbs", DoubleType, nullable),
      StructField("time to sixty", DoubleType, nullable),
      StructField("year", IntegerType, nullable),
      StructField("brand", StringType, nullable),
      StructField("origin", StringType, nullable)
    ))
    val rows = cars.map(line => line.split(",").map(elem => elem.trim))
      .map(x => Row.fromSeq(Seq(x(0).toDouble, x(1).toInt, x(2).toInt, x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toInt, x(7), x(8))))
    //println(rows)

    sqlContext.createDataFrame(rows, schema)
  }
}