package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import MDLPDiscretizerSuite.CLEAN_SUFFIX
import TestHelper._

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

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df, Array("mpg"), "origin", 10)

    assertResult("16.1, 21.05, 30.95, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  /** Its an error if maxBins is less than 2 */
  test("Run MDLPD on single mpg column in cars data (maxBins = 2") {

    val df = readCarsData(sqlContext)
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

    val df = readCarsData(sqlContext)
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

    val df = readCarsData(sqlContext)
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

}