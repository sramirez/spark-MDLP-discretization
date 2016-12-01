package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.apache.spark.ml.feature.TestHelper._


/**
  * Test MDLP discretization on a larger dataset.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class MDLPDiscretizerBigSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }


  /** Splitting on a date column should not be an error *
  test("Run MDLPD on srvRequest40000 with date column as label (label = closed date, maxBins = 10, maxByPart = 100000)") {

    val df = readSvcRequests40000Data(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Unique Key",
        "X Coordinate (State Plane)", "Y Coordinate (State Plane)", "Latitude", "Longitude"),
      "Closed Date", maxBins = 10, maxByPart = 100000, stoppingCriterion = 0, minBinPercentage = 0.5)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 1.45602196E12, 1.45611135E12, 1.45641701E12, 1.45642855E12, 1.4564372E12, 1.45644808E12, 1.45647757E12, 1.45649959E12, 1.45650955E12, 1.45651453E12, 1.4565224E12, 1.45654704E12, 1.45673683E12, 1.45675518E12, 1.4567654E12, 1.45677615E12, 1.45679371E12, 1.45681626E12, 1.45684483E12, 1.45686371E12, 1.45687917E12, 1.45691011E12, 1.4569316E12, 1.45699635E12, 1.45700684E12, 1.4570244E12, 1.45704354E12, 1.45708286E12, 1.45711405E12, 1.45711982E12, 1.45716701E12, 1.4571712E12, 1.45720056E12, 1.45722678E12, 1.45728052E12, 1.45730961E12, 1.4573222E12, 1.45734251E12, 1.45742771E12, 1.45743885E12, 1.45745183E12, Infinity;
        |-Infinity, 970456.5, 979403.5, 981275.5, 991150.5, 994490.0, 996590.5, 997077.5, 998123.0, 998398.5, 999451.5, 1000563.5, 1001776.0, 1002918.5, 1003944.5, 1004384.0, 1004505.5, 1007262.5, 1008990.0, 1010339.5, 1013073.0, 1013876.5, 1018213.0, 1018448.5, 1020292.0, 1021113.0, 1021618.5, 1022342.5, 1023333.5, 1026862.5, 1032483.5, 1035048.5, 1042767.0, 1044203.5, Infinity;
        |-Infinity, 147927.5, 150303.0, 155430.0, 156562.5, 158145.5, 161622.0, 172141.0, 175366.5, 179003.5, 186795.0, 190705.5, 195083.0, 196383.5, 199703.5, 202225.5, 207899.5, 216086.0, 222357.5, 227028.5, 229116.5, 231557.0, 234977.0, 237980.0, 239589.5, 241073.5, 241386.0, 253270.0, 253987.0, 255966.0, 258988.5, Infinity;
        |-Infinity, 40.572594, 40.57912, 40.593052, 40.59643, 40.60051, 40.60902, 40.639122, 40.647987, 40.657852, 40.679405, 40.690117, 40.70105, 40.70397, 40.70574, 40.714813, 40.721725, 40.73729, 40.75975, 40.782127, 40.78968, 40.795425, 40.802223, 40.8115, 40.81603, 40.819855, 40.82424, 40.82807, 40.82921, 40.86182, 40.86377, 40.869225, 40.87751, Infinity;
        |-Infinity, -74.04968, -74.01747, -74.01138, -73.97537, -73.96303, -73.95543, -73.954926, -73.94989, -73.94113, -73.93678, -73.93254, -73.928986, -73.92734, -73.92691, -73.916916, -73.91089, -73.90918, -73.87729, -73.87616, -73.86957, -73.86656, -73.865555, -73.86261, -73.8591, -73.846176, -73.82613, -73.81647, -73.788475, -73.78334, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }*/

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
        |-Infinity, 970456.5, 979403.5, 981275.5, 991150.5, 994490.0, 996590.5, 997077.5, 998123.0, 998398.5, 999451.5, 1000563.5, 1001776.0, 1002918.5, 1003944.5, 1004384.0, 1004505.5, 1007262.5, 1008990.0, 1010339.5, 1013073.0, 1013876.5, 1018213.0, 1018448.5, 1020292.0, 1021113.0, 1021618.5, 1022342.5, 1023333.5, 1026862.5, 1032483.5, 1035048.5, 1042767.0, 1044203.5, Infinity;
        |-Infinity, 147927.5, 150303.0, 155430.0, 156562.5, 158145.5, 161622.0, 172141.0, 175366.5, 179003.5, 186795.0, 190705.5, 195083.0, 196383.5, 199703.5, 202225.5, 207899.5, 216086.0, 222357.5, 227028.5, 229116.5, 231557.0, 234977.0, 237980.0, 239589.5, 241073.5, 241386.0, 253270.0, 253987.0, 255966.0, 258988.5, Infinity;
        |-Infinity, 40.572594, 40.57912, 40.593052, 40.59643, 40.60051, 40.60902, 40.639122, 40.647987, 40.657852, 40.679405, 40.690117, 40.70105, 40.70397, 40.70574, 40.714813, 40.721725, 40.73729, 40.75975, 40.782127, 40.78968, 40.795425, 40.802223, 40.8115, 40.81603, 40.819855, 40.82424, 40.82807, 40.82921, 40.86182, 40.86377, 40.869225, 40.87751, Infinity;
        |-Infinity, -74.04968, -74.01747, -74.01138, -73.97537, -73.96303, -73.95543, -73.954926, -73.94989, -73.94113, -73.93678, -73.93254, -73.928986, -73.92734, -73.92691, -73.916916, -73.91089, -73.90918, -73.87729, -73.87616, -73.86957, -73.86656, -73.865555, -73.86261, -73.8591, -73.846176, -73.82613, -73.81647, -73.788475, -73.78334, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

  /** Lots of rows (40k) few bins. Sometimes fails due to #14 */
  test("Run MDLPD on all columns in srvRequest40000 data (label = churned, maxBins = 10, maxByPart = 10000)") {

    val df = readSvcRequests40000Data(sqlContext)
    val model = getDiscretizerModel(df,
      Array("Unique Key", "Closed Date",
        "X Coordinate (State Plane)", "Y Coordinate (State Plane)", "Latitude", "Longitude"),
      "Borough", maxBins = 10, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 1.0)

    assertResult(
      """-Infinity, Infinity;
        |-Infinity, 1.45611057E12, 1.45654704E12, 1.45699635E12, 1.4570265E12, 1.45708286E12, 1.45716701E12, 1.45742234E12, 1.45743885E12, 1.45745825E12, Infinity;
        |-Infinity, 970456.5, 979403.5, 1000563.5, 1002918.5, 1004505.5, 1007262.5, 1008990.0, 1026862.5, 1035048.5, Infinity;
        |-Infinity, 147927.5, 161622.0, 175366.5, 186795.0, 190705.5, 196383.5, 207899.5, 231557.0, 255966.0, Infinity;
        |-Infinity, 40.572594, 40.60902, 40.647987, 40.679405, 40.690117, 40.70574, 40.73729, 40.802223, 40.869225, Infinity;
        |-Infinity, -74.04968, -74.01747, -73.94113, -73.93254, -73.92691, -73.916916, -73.90918, -73.846176, -73.81647, Infinity
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
        |-Infinity, 970456.5, 979403.5, 1000563.5, 1002918.5, 1004505.5, 1007262.5, 1008990.0, 1026862.5, 1035048.5, Infinity;
        |-Infinity, 147927.5, 161622.0, 175366.5, 186795.0, 190705.5, 196383.5, 207899.5, 231557.0, 255966.0, Infinity;
        |-Infinity, 40.572594, 40.60902, 40.647987, 40.679405, 40.690117, 40.70574, 40.73729, 40.802223, 40.869225, Infinity;
        |-Infinity, -74.04968, -74.01747, -73.94113, -73.93254, -73.92691, -73.916916, -73.90918, -73.846176, -73.81647, Infinity
        |""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

}