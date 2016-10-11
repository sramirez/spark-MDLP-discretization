package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._


/**
  * Test MDLP discretization
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class MDLPDiscretizerSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = _

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
        |-Infinity, 2379.5, 2959.5, 3412.5, Infinity;
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
        |-Infinity, 97.5, 106.0, 120.5, 134.5, 140.5, 148.5, 159.5, 169.5, Infinity;
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
        |-Infinity, Infinity
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
        |-Infinity, 6.6229, 7.175, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.795851, 74.375, Infinity;
        |-Infinity, 1.5, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /**
    * Simulate big data by lowering the maxByPart value to 100.
    * Sometimes fails due to issue #14.
    */
  test("Run MDLPD on all columns in titanic data (label = embarked, maxByPart = 100)") {

    val df = readTitanicData(sqlContext)
    val model = getDiscretizerModel(df, Array("age", "fare", "pclass", "sibsp", "parch", "grad date"),
      "embarked", maxBins = 100, maxByPart = 100)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 6.6229, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 29.4125, 74.375, Infinity;
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
        |-Infinity, 6.6229, 7.1833496, 7.2396, 7.6875, 7.7625, 13.20835, 15.3729, 15.74585, 56.7125, Infinity;
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

  test("Run MDLPD on columns in titanic data with double nulls converted to NaN (label = parch)") {

    val df = cleanNumericCols(readTitanicData(sqlContext), Array("age", "fare"))

    //df.select("age" + CLEAN_SUFFIX, "fare" + CLEAN_SUFFIX).show(100)
    val model = getDiscretizerModel(df, Array("age" + CLEAN_SUFFIX, "fare" + CLEAN_SUFFIX), "parch")

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, Infinity
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
        |-Infinity, 13.825001, 29.0625, 44.65, 51.67085, 152.50626, Infinity;
        |-Infinity, 2.5, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 1.44359817E12, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in iris data (label = iristype)") {

    val df = readIrisData(sqlContext)
    val model = getDiscretizerModel(df, Array("sepallength", "sepalwidth", "petallength", "petalwidth"), "iristype")

    assertResult(
      """-Infinity, 5.55, Infinity;
        |-Infinity, 3.35, Infinity;
        |-Infinity, 2.45, 4.75, Infinity;
        |-Infinity, 0.8, 1.75, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  test("Run MDLPD on all columns in iris data (label = iristype, splittingCriterion = -0.01)") {

    val df = readIrisData(sqlContext)
    val model = getDiscretizerModel(df, Array("sepallength", "sepalwidth", "petallength", "petalwidth"),
      "iristype", maxBins = 100, maxByPart = 10000, stoppingCriterion = -0.01)

    assertResult(
      """-Infinity, 5.55, 6.1499996, Infinity;
        |-Infinity, 2.95, 3.35, Infinity;
        |-Infinity, 2.45, 4.75, Infinity;
        |-Infinity, 0.8, 1.75, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /**
    * In this case do not convert nulls to a special MISSING value.
    * An error should be reported if the label column contains NaN.
    */
  test("Run MDLPD on null_label_test data with nulls in label") {
    val df = readNullLabelTestData(sqlContext)

    try {
      createDiscretizerModel(df, Array("col1", "col2"), "label")
      fail("there should have been an exception")
    }
    catch {
      case ex: IllegalArgumentException =>
        assert(ex.getMessage.startsWith("Some NaN values have been found") , "Unexpected message: " + ex.getMessage)
      case otherEx : Throwable => fail("Unexpected error: " + otherEx)
    }
  }

  /** the min bin percent is 5% */
  test("Run MDLPD on all columns in churn data (label = churned, maxBins = 1000, minBinPercentage = 5.0)") {
    val df = readChurnData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Number Vmail Messages", "Total Day Minutes", "Total Day Calls", "Total Day Charge", "Total Eve Minutes",
        "Calls", "Charge", "Total Night Minutes", "Total Night Calls", "Total Night Charge", "Total Intl Minutes",
        "Total Intl Calls", "Total Intl Charge", "Number Customer Service Calls"),
      "Churned", maxBins = 1000, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 5.0)

    assertResult(
      """-Infinity, 2.0, Infinity;
        |-Infinity, 168.05, 221.85, 248.65, 271.05, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 28.57, 37.715, 42.269997, 46.08, Infinity;
        |-Infinity, 248.15, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 21.095001, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 13.15, Infinity;
        |-Infinity, 2.5, Infinity;
        |-Infinity, 3.55, Infinity;
        |-Infinity, 3.5, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** More bins will be generated because the minBinPercentage is lower. */
  test("Run MDLPD on all columns in churn data (label = churned, maxBins = 1000, minBinPercentage = 0.01)") {

    val df = readChurnData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Number Vmail Messages", "Total Day Minutes", "Total Day Calls", "Total Day Charge", "Total Eve Minutes",
        "Calls", "Charge", "Total Night Minutes", "Total Night Calls", "Total Night Charge", "Total Intl Minutes",
        "Total Intl Calls", "Total Intl Charge", "Number Customer Service Calls"),
      "Churned", maxBins = 1000, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 0.01)

    assertResult(
      """-Infinity, 2.0, Infinity;
        |-Infinity, 168.05, 221.85, 248.65, 285.5, 316.35, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 28.57, 37.715, 42.269997, 48.535, 53.78, Infinity;
        |-Infinity, 248.15, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 21.095001, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, 13.15, Infinity;
        |-Infinity, 2.5, Infinity;
        |-Infinity, 3.55, Infinity;
        |-Infinity, 3.5, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** state is a label with many values and none of them correlate well with the other columns */
  test("Run MDLPD on all columns in churn data (label = state, maxBins = 1000)") {

    val df = readChurnData(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Number Vmail Messages", "Total Day Minutes", "Total Day Calls", "Total Day Charge", "Total Eve Minutes",
        "Calls", "Charge", "Total Night Minutes", "Total Night Calls", "Total Night Charge", "Total Intl Minutes",
        "Total Intl Calls", "Total Intl Charge", "Number Customer Service Calls"),
      "State", maxBins = 1000)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity;
        |-Infinity, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

}