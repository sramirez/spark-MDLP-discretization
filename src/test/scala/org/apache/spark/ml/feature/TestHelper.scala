package org.apache.spark.ml.feature

import java.sql.Timestamp

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.joda.time.format.DateTimeFormat

/**
  * Loads various test datasets
  */
object TestHelper {

  final val SPARK_CTX = createSparkContext()
  final val FILE_PREFIX = "src/test/resources/data/"
  final val ISO_DATE_FORMAT = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss")
  final val NULL_VALUE = "?"


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

  /** @return the titanic data as a dataframe. This dataset has nulls and dates */
  def readTitanicData(sqlContext: SQLContext): DataFrame = {
    val titanic = SPARK_CTX.textFile(FILE_PREFIX + "titanic.data")
    val nullable = true

    // No,       Braund,male,22,  A/5 21171,7.25, ?,     3,      S,        1,     0,     2015-04-22T00:00:00
    val schema = StructType(List(
      StructField("survived", StringType, nullable),
      StructField("name", StringType, nullable),
      StructField("sex", StringType, nullable),
      StructField("age", DoubleType, nullable),
      StructField("ticket", StringType, nullable),
      StructField("fare", DoubleType, nullable),
      StructField("cabin", StringType, nullable),
      StructField("pclass", DoubleType, nullable), // int
      StructField("embarked", StringType, nullable),
      StructField("sibsp", DoubleType, nullable), // int
      StructField("parch", DoubleType, nullable), // int
      StructField("grad date", DoubleType, nullable)
    ))
    // ints and dates must be read as doubles
    val rows = titanic.map(line => line.split(",").map(elem => elem.trim))
      .map(x => Row.fromSeq(Seq(
        asString(x(0)), asString(x(1)), asString(x(2)),
        asDouble(x(3)), asString(x(4)), asDouble(x(5)), asString(x(6)),
        asDouble(x(7)), asString(x(8)), asDouble(x(9)), asDouble(x(10)), asDateDouble(x(11))
        )
      ))

    sqlContext.createDataFrame(rows, schema)
  }


  /** @return the titanic data as a dataframe. This version is interesting because the VectorAssembler
    *         makes some of its values sparse and other dense. In the other version they are all dense.
    */
  def readTitanic2Data(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "titanic2.data")
    val nullable = true

    // No	?	3	S	1	0
    val schema = StructType(List(
      StructField("survived", StringType, nullable),
      StructField("name", StringType, nullable),
      StructField("sex", StringType, nullable),
      StructField("age", DoubleType, nullable),
      StructField("ticket", StringType, nullable),
      StructField("fare", DoubleType, nullable),
      StructField("cabin", StringType, nullable),
      StructField("pclass", DoubleType, nullable), // int
      StructField("embarked", StringType, nullable),
      StructField("sibsp", DoubleType, nullable), // int
      StructField("parch", DoubleType, nullable)  // int
    ))
    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {
        Row.fromSeq(Seq(
          asString(x(0)), asString(x(1)), asString(x(2)), asDouble(x(3)), asString(x(4)), asDouble(x(5)),
          asString(x(6)), asDouble(x(7)), asString(x(8)), asDouble(x(9)), asDouble(x(10))))
      })

    sqlContext.createDataFrame(rows, schema)
  }


  /** @return standard iris dataset from UCI repo.
    */
  def readIrisData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "iris.data")
    val nullable = true

    val schema = StructType(List(
      StructField("sepallength", DoubleType, nullable),
      StructField("sepalwidth", DoubleType, nullable),
      StructField("petallength", DoubleType, nullable),
      StructField("petalwidth", DoubleType, nullable),
      StructField("iristype", StringType, nullable)
    ))
    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asDouble(x(0)), asDouble(x(1)), asDouble(x(2)), asDouble(x(3)), asString(x(4))))})

    sqlContext.createDataFrame(rows, schema)
  }


  /** @return dataset with 3 double columns. The first is the label column and contain null.
    */
  def readNullLabelTestData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "null_label_test.data")
    val nullable = true

    val schema = StructType(List(
      StructField("label_IDX", DoubleType, nullable),
      StructField("col1", DoubleType, nullable),
      StructField("col2", DoubleType, nullable)
    ))
    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asDouble(x(0)), asDouble(x(1)), asDouble(x(2))))})

    sqlContext.createDataFrame(rows, schema)
  }

  private def asDateDouble(isoString: String) = {
    if (isoString == NULL_VALUE) Double.NaN
    else ISO_DATE_FORMAT.parseDateTime(isoString).getMillis.toString.toDouble
  }

  // label cannot currently have null values - see #8.
  private def asString(value: String) = if (value == NULL_VALUE) null else value
  private def asDouble(value: String) = if (value == NULL_VALUE) Double.NaN else value.toDouble
}
