package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import MDLPDiscretizerSuite.CLEAN_SUFFIX
import TestHelper._

object MDLPDiscretizerSuite {
  val CLEAN_SUFFIX: String = "_CLEAN"
}

/**
  * Test MDLP discretization
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
  test("Run MDLPD on single mpg column in cars data (maxBins = 10)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df, Array("mpg"), "origin", 10)

    assertResult("-Infinity, 16.1, 21.05, 30.95, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  /** Its an error if maxBins is less than 2 */
  test("Run MDLPD on single mpg column in cars data (maxBins = 2)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df, Array("mpg"), "origin", 2)

    assertResult("-Infinity, 21.05, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  /**
    * Do entropy based binning of cars data for all the numeric columns using origin as target.
    * The algorithm may produce different splits for individual columns when several features are discretized
    * at once because of the interaction between features.
    */
  test("Run MDLPD on all columns in cars data (maxBins = 100, label = origin)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "origin", 100)

    assertResult(
      """-Infinity, 16.1, 21.05, 30.95, Infinity;
        |-Infinity, 5.5, Infinity;
        |-Infinity, 97.5, 169.5, Infinity;
        |-Infinity, 78.5, 134.0, Infinity;
        |-Infinity, 2379.5, 2959.5, Infinity;
        |-Infinity, 13.5, 19.5, Infinity;
        |-Infinity, 1980.5, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in cars data (maxBins = 100, label = brand)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "brand", 100)

    assertResult(
      """-Infinity, 21.05, Infinity;
        |-Infinity, 5.5, Infinity;
        |-Infinity, 120.5, 134.5, Infinity;
        |-Infinity, 78.5, Infinity;
        |-Infinity, 2550.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** Do entropy based binning of cars data from UC Irvine repository. */
  test("Run MDLPD on single age column in titanic data (label = pclass)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age"), "pclass")

    assertResult("-Infinity, 34.75, Infinity") {
      model.splits(0).mkString(", ")
    }
  }

  test("Run MDLPD on all columns in titanic data (label = survived)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "survived")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 10.48125, 74.375, Infinity;
        |-Infinity, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 1.44359817E12, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
        model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /**
    * If the label has actuall null values, this throws an NPE.
    * Nulls are currently represented with "?"
    */
  test("Run MDLPD on all columns in titanic data (label = embarked)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "embarked")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 6.6229, 6.9624996, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 56.7125, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in titanic data (label = sex)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "sex")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 9.54375, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 0.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 1.44359817E12, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /**
    * Aparently, all numeric columns are fairly random with respect to cabin.
    */
  test("Run MDLPD on all columns in titanic data (label = cabin)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "cabin")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  // here, the label is continuous
  test("Run MDLPD on all columns in titanic data (label = sibsp)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "parch", "grad date"), "sibsp")

    assertResult(
      """-Infinity, 14.75, Infinity;
        |-Infinity, 13.825001, 28.2, 41.9896, 44.65, 47.0, 51.67085, 152.50626, Infinity;
        |-Infinity, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 1.44359817E12, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /**
    * @return the discretizer fit to the data given the specified features to bin and label use as target.
    */
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

    discretizer.fit(processedDf)
  }

}