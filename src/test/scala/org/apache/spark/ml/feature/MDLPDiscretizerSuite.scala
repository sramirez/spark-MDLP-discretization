package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import MDLPDiscretizerSuite._
import TestHelper._
import org.apache.spark.sql.functions.{when, lit, col}

object MDLPDiscretizerSuite {

  // This value is used to represent nulls in string columns
  val MISSING = "__MISSING_VALUE__"

  val CLEAN_SUFFIX: String = "_CLEAN"
  val INDEX_SUFFIX: String = "_IDX"
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

  /** Do entropy based binning of cars data for all the numeric columns using origin as target. */
  test("Run MDLPD on all columns in cars data (label = origin, maxBins = 100)") {

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

  /** Simulate big data by lowering the maxByPart value to 100. */
  test("Run MDLPD on all columns in cars data (label = origin, maxBins = 100, maxByPart = 100)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "origin", maxBins = 100, maxByPart = 100)

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

  /** Lowering the stopping criterion should result in more splits */
  test("Run MDLPD on all columns in cars data (label = origin, maxBins = 100, stoppingCriterion = -1e-2)") {

    val df = readCarsData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("mpg", "cylinders", "cubicinches", "horsepower", "weightlbs", "time to sixty", "year"),
      "origin", maxBins = 100, maxByPart = 10000, stoppingCriterion = -1e-2)

    assertResult(
      """-Infinity, 16.1, 21.05, 30.95, Infinity;
        |-Infinity, 5.5, 7.0, Infinity;
        |-Infinity, 97.5, 106.0, 120.5, 134.5, 140.5, 148.5, 159.5, 165.5, 169.5, Infinity;
        |-Infinity, 78.5, 134.0, Infinity;
        |-Infinity, 2379.5, 2959.5, 3274.0, Infinity;
        |-Infinity, 13.5, 19.5, Infinity;
        |-Infinity, 1980.5, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in cars data (label = brand, maxBins = 100)") {

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
    * If the label has actual null values, this throws an NPE.
    * Nulls are currently represented with "?"
    */
  test("Run MDLPD on all columns in titanic data (label = embarked)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "embarked")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 6.6229, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 56.7125, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }


  /**
    * The discretization of the parch column at one time did not work because the
    * feature vector had a mix of dense and sparse vectors when the VectorAssembler was
    * applied to this dataset.
    */
  test("Run MDLPD on all columns in titanic2 data (label = embarked)") {

    val df = readTitanic2Data(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch"), "embarked")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 7.175, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.795851, 74.375, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** Simulate big data by lowering the maxByPart value to 100. */
  test("Run MDLPD on all columns in titanic data (label = embarked, maxByPart = 100)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"),
      "embarked", maxBins = 100, maxByPart = 100)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 6.6229, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 56.7125, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** lowering the stoppingCriterion should result in more splits. */
  test("Run MDLPD on all columns in titanic data (label = embarked, stoppingCriterion = -1e-2)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"), "embarked",
      maxBins = 100, maxByPart = 10000, stoppingCriterion = -1e-2)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 6.6229, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 29.4125, 56.7125, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, 2.5, Infinity;
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

  /** Simulate big data by lowering the maxByPart value to 100. */
  test("Run MDLPD on all columns in titanic data (label = sex, maxByPart = 100)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"),
      "sex", maxBins = 100, maxByPart = 100)

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
    * Apparently, all numeric columns are fairly random with respect to cabin.
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
  def getDiscretizerModel(dataframe: DataFrame, inputCols: Array[String],
                          labelColumn: String,
                          maxBins: Int = 100,
                          maxByPart: Int = 10000,
                          stoppingCriterion: Double = 0): DiscretizerModel = {
    val df = dataframe
      .withColumn(labelColumn + CLEAN_SUFFIX, when(col(labelColumn).isNull, lit(MISSING)).otherwise(col(labelColumn)))

    val labelIndexer = new StringIndexer()
      .setInputCol(labelColumn + CLEAN_SUFFIX)
      .setOutputCol(labelColumn + INDEX_SUFFIX).fit(df)

    var processedDf = labelIndexer.transform(df)

    val featureAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
    processedDf = featureAssembler.transform(processedDf)
    //processedDf.select("features").show(800)

    val discretizer = new MDLPDiscretizer()
      .setMaxBins(maxBins)
      .setMaxByPart(maxByPart)
      .setStoppingCriterion(stoppingCriterion)
      .setInputCol("features")  // this must be a feature vector
      .setLabelCol(labelColumn + INDEX_SUFFIX)
      .setOutputCol("bucketFeatures")

    discretizer.fit(processedDf)
  }

}