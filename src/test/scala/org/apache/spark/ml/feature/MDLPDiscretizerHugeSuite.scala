package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.TestHelper._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}


/**
  * Test MDLP discretization on a larger dataset.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class MDLPDiscretizerHugeSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }

  /*
   If a continuous target with many values is used, this will result in
   ERROR Executor: Managed memory leak detected; size = 2692467336 bytes
   See https://issues.apache.org/jira/browse/SPARK-18181
   Since the serverX_10000000.data data file is large, I did not check it into git.
   It (zipped version) can be retrieved from https://app.box.com/s/mrepegkdjv7im0iq1f97slp8c88h6rwj
   if you want to run this test using it.
   */
  /* Commented because it takes a long time, and the data needs to be downloaded separately
  test("Run MDLPD on all columns in serverBigX data (label = targetB, maxBins = 50, maxByPart = 100000)") {
    val dataDf = readServerBigXData(sqlContext)
    val model = getDiscretizerModel(dataDf,
      Array("val3", "val4", "val5", "val6"),
      "targetB", maxBins = 50, maxByPart = 100000, stoppingCriterion = 0, minBinPercentage = 1)

    assertResult(
      """-Infinity, 37.5, 42.5, 47.5, Infinity;
        |-Infinity, 203.125, 281.875, 360.625, Infinity;
        |-Infinity, 337.5539, 363.06793, Infinity;
        |-Infinity, 64.4039, 89.91792, Infinity"""
        .stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }*/

  test("Run MDLPD on all columns in serverX data (label = target2, maxBins = 50, maxByPart = 10000)") {
    val dataDf = readServerXData(sqlContext)
    val model = getDiscretizerModel(dataDf,
      Array("CPU1_TJ", "CPU2_TJ", "total_cfm", "rpm1"),
      "target4", maxBins = 50, maxByPart = 10000, stoppingCriterion = 0, minBinPercentage = 1)

    assertResult(
      """-Infinity, 337.55365, 363.06793, Infinity;
        |-Infinity, 329.35974, 330.47424, 331.16617, 331.54724, 332.8419, 333.82208, 334.7564, 335.65106, 336.6503, 337.26328, 337.8406, 339.16763, 339.81476, 341.1809, 341.81186, 343.64825, 355.91144, 357.8602, 361.57806, Infinity;
        |-Infinity, 0.0041902177, 0.0066683707, 0.00841628, 0.009734755, 0.011627266, 0.012141651, 0.012740928, 0.013055362, 0.013293093, 0.014488807, 0.014869433, 0.015116488, 0.015383363, 0.015662778, 0.015978532, 0.016246023, 0.016492717, 0.01686273, 0.017246526, 0.017485093, 0.017720722, 0.017845878, 0.018008012, 0.018357705, 0.018629191, 0.018964633, 0.019226547, 0.019445801, 0.01960973, 0.019857172, 0.020095222, 0.020373512, 0.020728927, 0.020977266, 0.02137091, 0.021543117, 0.02188059, 0.022238541, 0.02265025, 0.023091711, 0.023352059, 0.023588676, 0.023957964, 0.024230447, 0.024448851, 0.024822969, 0.025079254, 0.026178652, 0.027195029, Infinity;
        |-Infinity, 1500.0, 4500.0, 7500.0, Infinity""".stripMargin.replaceAll(System.lineSeparator(), "")) {
      model.splits.map(a => a.mkString(", ")).mkString(";")
    }
  }

}