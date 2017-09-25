package org.apache.spark.ml.feature

import org.apache.spark.mllib.feature.DiscretizationUtils._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}


/**
  * Test entropy calculation.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class FeatureUtilsSuite extends FunSuite {

  test("Test entropy calc (typical 1)") {

    val frequencies = Seq(100L, 100L, 200L)

    assertResult(1.5) {
      entropy(frequencies, 400)
    }
  }

  // The maximum entropy is log base 2 of 3 = 1.585.
  test("Test entropy calc all same (maximum)") {

    val frequencies = Seq(100L, 100L, 100L)

    assertResult(1.584962500721156) {
      entropy(frequencies, 300)
    }
  }

  test("Test entropy calc all different (typical 2)") {

    val frequencies = Seq(100L, 0L, 200L)

    assertResult(0.9182958340544896) {
      entropy(frequencies, 300)
    }
  }

  test("Test entropy calc all different (typical 3)") {

    val frequencies = Seq(10L, 0L, 290L)

    assertResult(0.21084230031853213) {
      entropy(frequencies, 300)
    }
  }

  test("Test entropy calc all different (typical 4)") {

    val frequencies = Seq(5L, 5L, 290L)

    assertResult(0.24417563365186545) {
      entropy(frequencies, 300)
    }
  }

  test("Test entropy calc all different (minimum)") {

    val frequencies = Seq(0L, 0L, 300L)

    assertResult(0) {
      entropy(frequencies, 300)
    }
  }


}