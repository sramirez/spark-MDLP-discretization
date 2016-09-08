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
        |-Infinity, 13.825001, 28.2, 41.9896, 47.0, 51.67085, 152.50626, Infinity;
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
      """-Infinity, 5.55, 6.1499996, Infinity;
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
        |-Infinity, 2.45, 4.75, 5.1499996, Infinity;
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

  /** Lots of rows (40k) many bins. This will result in a lot of splits */
  test("Run MDLPD on all columns in srvRequest40000 data (label = churned, maxBins = 100, maxByPart = 10000)") {

    val df = readSvcRequests40000Data(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Unique Key", "Closed Date",
        "X Coordinate (State Plane)", "Y Coordinate (State Plane)", "Latitude", "Longitude"),
      "Borough", maxBins = 100, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 0.5)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 1.45602196E12, 1.45611135E12, 1.45641701E12, 1.45642855E12, 1.4564372E12, 1.45644808E12, 1.45647757E12, 1.45649959E12, 1.45650955E12, 1.45651453E12, 1.4565224E12, 1.45654704E12, 1.45673683E12, 1.45675518E12, 1.4567654E12, 1.45677615E12, 1.45679371E12, 1.45681626E12, 1.45684483E12, 1.45686371E12, 1.45687917E12, 1.45691011E12, 1.4569316E12, 1.45699635E12, 1.45700684E12, 1.4570244E12, 1.45704354E12, 1.45708286E12, 1.45711405E12, 1.45711982E12, 1.45716701E12, 1.4571712E12, 1.45720056E12, 1.45722678E12, 1.45728052E12, 1.45730961E12, 1.4573222E12, 1.45734251E12, 1.45742771E12, 1.45743885E12, 1.45745183E12, Infinity;
        |-Infinity, 970456.5, 979403.5, 981275.5, 991150.5, 994490.0, 996590.5, 997077.5, 998123.0, 998398.5, 999451.5, 1000563.5, 1001776.0, 1002918.5, 1003944.5, 1004384.0, 1004505.5, 1007262.5, 1008990.0, 1010339.5, 1013073.0, 1013876.5, 1018213.0, 1018448.5, 1020292.0, 1021113.0, 1021618.5, 1022342.5, 1023333.5, 1026816.5, 1032483.5, 1035048.5, 1042767.0, 1044203.5, Infinity;
        |-Infinity, 147927.5, 150303.0, 155430.0, 156562.5, 158145.5, 161622.0, 172141.0, 175366.5, 178198.5, 183054.5, 187831.5, 190705.5, 192939.5, 195410.5, 196034.5, 199703.5, 202225.5, 207899.5, 216086.0, 222357.5, 227028.5, 229116.5, 231557.0, 234977.0, 237980.0, 239589.5, 241073.5, 241386.0, 253270.0, 253987.0, 255966.0, Infinity;
        |-Infinity, 40.572594, 40.57912, 40.593052, 40.59643, 40.60051, 40.60902, 40.639122, 40.647987, 40.65739, 40.669113, 40.6821, 40.690117, 40.695152, 40.70105, 40.704575, 40.714813, 40.721725, 40.73729, 40.75975, 40.782127, 40.78968, 40.795425, 40.802223, 40.8115, 40.81603, 40.819855, 40.82424, 40.82807, 40.82921, 40.86182, 40.86377, 40.869225, Infinity;
        |-Infinity, -74.04968, -74.01747, -74.01138, -73.97537, -73.96303, -73.95543, -73.954926, -73.94989, -73.94113, -73.93678, -73.93254, -73.928986, -73.92734, -73.92691, -73.916916, -73.910706, -73.905655, -73.89888, -73.87729, -73.87616, -73.86733, -73.865555, -73.86261, -73.860245, -73.846176, -73.82417, -73.81647, -73.788475, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** Lots of rows (40k) few bins. */
  test("Run MDLPD on all columns in srvRequest40000 data (label = churned, maxBins = 10, maxByPart = 10000)") {

    val df = readSvcRequests40000Data(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Unique Key", "Closed Date",
        "X Coordinate (State Plane)", "Y Coordinate (State Plane)", "Latitude", "Longitude"),
      "Borough", maxBins = 10, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 1.0)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 1.45611057E12, 1.45654704E12, 1.45699635E12, 1.4570265E12, 1.45708286E12, 1.45716701E12, 1.45742234E12, 1.45743885E12, 1.45745825E12, Infinity;
        |-Infinity, 970456.5, 979403.5, 1000563.5, 1002918.5, 1004505.5, 1007262.5, 1008990.0, 1022251.5, 1032483.5, Infinity;
        |-Infinity, 147927.5, 161622.0, 175366.5, 183054.5, 187831.5, 196034.5, 207899.5, 231557.0, 239589.5, Infinity;
        |-Infinity, 40.572594, 40.60902, 40.647987, 40.669113, 40.6821, 40.704575, 40.73729, 40.802223, 40.82424, Infinity;
        |-Infinity, -74.04968, -74.01747, -73.94113, -73.93254, -73.92691, -73.916916, -73.910706, -73.86261, -73.82417, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** Lots of rows (40k) few bins. */
  test("Run MDLPD on all columns in srvRequest40000 data (label = churned, maxBins = 10, maxByPart = 1000000)") {

    val df = readSvcRequests40000Data(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Unique Key", "Closed Date",
        "X Coordinate (State Plane)", "Y Coordinate (State Plane)", "Latitude", "Longitude"),
      "Borough", maxBins = 10, maxByPart = 1000000, stoppingCriterion = 0, minBinPercentage = 1.0)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 1.45611057E12, 1.45654704E12, 1.45699635E12, 1.4570265E12, 1.45708286E12, 1.45716701E12, 1.45742234E12, 1.45743885E12, 1.45745825E12, Infinity;
        |-Infinity, 970456.5, 979403.5, 1000563.5, 1002918.5, 1004505.5, 1007262.5, 1008990.0, 1022251.5, 1032483.5, Infinity;
        |-Infinity, 147927.5, 161622.0, 175366.5, 183054.5, 187831.5, 196034.5, 207899.5, 231557.0, 239589.5, Infinity;
        |-Infinity, 40.572594, 40.60902, 40.647987, 40.669113, 40.6821, 40.704575, 40.73729, 40.802223, 40.82424, Infinity;
        |-Infinity, -74.04968, -74.01747, -73.94113, -73.93254, -73.92691, -73.916916, -73.910706, -73.86261, -73.82417, Infinity
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

  /**
    * @return the discretizer fit to the data given the specified features to bin and label use as target.
    */
  def getDiscretizerModel(dataframe: DataFrame, inputCols: Array[String],
                          labelColumn: String,
                          maxBins: Int = 100,
                          maxByPart: Int = 10000,
                          stoppingCriterion: Double = 0,
                          minBinPercentage: Double = 0): DiscretizerModel = {
    val processedDf = cleanLabelCol(dataframe, labelColumn)
    createDiscretizerModel(processedDf, inputCols, labelColumn, maxBins, maxByPart, stoppingCriterion, minBinPercentage)
  }

  def createDiscretizerModel(dataframe: DataFrame, inputCols: Array[String],
                             labelColumn: String,
                             maxBins: Int = 100,
                             maxByPart: Int = 10000,
                             stoppingCriterion: Double = 0,
                             minBinPercentage: Double = 0): DiscretizerModel = {
    val featureAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
    val processedDf = featureAssembler.transform(dataframe)

    val discretizer = new MDLPDiscretizer()
      .setMaxBins(maxBins)
      .setMaxByPart(maxByPart)
      .setStoppingCriterion(stoppingCriterion)
      .setMinBinPercentage(minBinPercentage)
      .setInputCol("features") // this must be a feature vector
      .setLabelCol(labelColumn + INDEX_SUFFIX)
      .setOutputCol("bucketFeatures")

    discretizer.fit(processedDf)
  }

  def cleanLabelCol(dataframe: DataFrame, labelColumn: String): DataFrame = {
    val df = dataframe
      .withColumn(labelColumn + CLEAN_SUFFIX, when(col(labelColumn).isNull, lit(MISSING)).otherwise(col(labelColumn)))

    convertLabelToIndex(df, labelColumn + CLEAN_SUFFIX, labelColumn + INDEX_SUFFIX)
  }

  def convertLabelToIndex(df: DataFrame, inputCol: String, outputCol: String): DataFrame = {

    val labelIndexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol).fit(df)

    labelIndexer.transform(df)
  }

}