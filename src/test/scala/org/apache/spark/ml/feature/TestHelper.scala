package org.apache.spark.ml.feature

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._

/**
  * Loads various test datasets
  */
object TestHelper {

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

  /** @return the cars data as a dataframe */
  def readCarsData(sqlContext: SQLContext): DataFrame = {
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

  /** @return the cars data as a dataframe */
  def readTitanicData(sqlContext: SQLContext): DataFrame = {
    val cars = SPARK_CTX.textFile(FILE_PREFIX + "titanic.data")
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
