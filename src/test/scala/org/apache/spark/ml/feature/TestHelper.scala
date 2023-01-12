package org.apache.spark.ml.feature

import java.sql.Timestamp

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.functions._
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

  // This value is used to represent nulls in string columns
  final val MISSING = "__MISSING_VALUE__"
  final val CLEAN_SUFFIX: String = "_CLEAN"
  final val INDEX_SUFFIX: String = "_IDX"

  /**
    * @return the discretizer fit to the data given the specified features to bin and label use as target.
    */
  def createDiscretizerModel(dataframe: DataFrame, inputCols: Array[String],
                             labelColumn: String,
                             maxBins: Int = 100,
                             maxByPart: Int = 10000,
                             stoppingCriterion: Double = 0,
                             minBinPercentage: Double = 0,
                             approximate: Boolean = false): DiscretizerModel = {
    val featureAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val processedDf = featureAssembler.transform(dataframe)

    val discretizer = new MDLPDiscretizer()
      .setMaxBins(maxBins)
      .setMaxByPart(maxByPart)
      .setStoppingCriterion(stoppingCriterion)
      .setMinBinPercentage(minBinPercentage)
      .setInputCol("features") // this must be a feature vector
      .setLabelCol(labelColumn + INDEX_SUFFIX)
      .setOutputCol("bucketFeatures")
      .setApproximate(approximate)

    discretizer.fit(processedDf)
  }


  /**
    * The label column will have null values replaced with MISSING values in this case.
    * @return the discretizer fit to the data given the specified features to bin and label use as target.
    */
  def getDiscretizerModel(dataframe: DataFrame, inputCols: Array[String],
                          labelColumn: String,
                          maxBins: Int = 100,
                          maxByPart: Int = 10000,
                          stoppingCriterion: Double = 0,
                          minBinPercentage: Double = 0, 
                          approximate: Boolean = false): DiscretizerModel = {
    val processedDf = cleanLabelCol(dataframe, labelColumn)
    createDiscretizerModel(processedDf, inputCols, labelColumn, 
        maxBins, maxByPart, stoppingCriterion, minBinPercentage, approximate)
  }


  def cleanLabelCol(dataframe: DataFrame, labelColumn: String): DataFrame = {
    val df = dataframe
      .withColumn(labelColumn + CLEAN_SUFFIX, when(col(labelColumn).isNull, lit(MISSING)).otherwise(col(labelColumn)))

    convertLabelToIndex(df, labelColumn + CLEAN_SUFFIX, labelColumn + INDEX_SUFFIX)
  }

  def cleanNumericCols(dataframe: DataFrame, numericCols: Array[String]): DataFrame = {
    var df = dataframe
    numericCols.foreach(column => {
      df = df.withColumn(column + CLEAN_SUFFIX, when(col(column).isNull, lit(Double.NaN)).otherwise(col(column)))
    })
    df
  }

  def convertLabelToIndex(df: DataFrame, inputCol: String, outputCol: String): DataFrame = {

    val labelIndexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol).fit(df)

    labelIndexer.transform(df)
  }

  def createSparkContext() = {
    // the [n] corresponds to the number of worker threads and should correspond ot the number of cores available.
    val conf = new SparkConf().setAppName("test-spark").setMaster("local[4]")
    // Changing the default parallelism to 4 hurt performance a lot for a big dataset.
    // When maxByPart was 10000, it wend from 39 min to 4.5 hours.
    //conf.set("spark.default.parallelism", "4")
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

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return the dates data as a dataframe. */
  def readDatesData(sqlContext: SQLContext): DataFrame = {
    val datesData = SPARK_CTX.textFile(FILE_PREFIX + "dates.data")
    val nullable = true

    // txt, date
    val schema = StructType(List(
      StructField("txt", StringType, nullable),
      StructField("date", DoubleType, nullable)
    ))
    val rows = datesData.map(line => line.split(",").map(elem => elem.trim))
      .map(x => Row.fromSeq(Seq(asString(x(0)), asDateDouble(x(1)))))

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

  /** @return subset of fake telecom churn dataset. This dataset has more rows than the others.
    */
  def readChurnData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "churn.data")
    val nullable = true

    val schema = StructType(List(
      StructField("State", StringType, nullable),
      StructField("Number Vmail Messages", DoubleType, nullable),
      StructField("Total Day Minutes", DoubleType, nullable),
      StructField("Total Day Calls", DoubleType, nullable),
      StructField("Total Day Charge", DoubleType, nullable),
      StructField("Total Eve Minutes", DoubleType, nullable),
      StructField("Calls", DoubleType, nullable),
      StructField("Charge", DoubleType, nullable),
      StructField("Total Night Minutes", DoubleType, nullable),
      StructField("Total Night Calls", DoubleType, nullable),
      StructField("Total Night Charge", DoubleType, nullable),
      StructField("Total Intl Minutes", DoubleType, nullable),
      StructField("Total Intl Calls", DoubleType, nullable),
      StructField("Total Intl Charge", DoubleType, nullable),
      StructField("Number Customer Service Calls", DoubleType, nullable),
      StructField("Churned", StringType, nullable)
    ))

    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asString(x(0)), asDouble(x(1)), asDouble(x(2)), asDouble(x(3)), asDouble(x(4)),
        asDouble(x(5)), asDouble(x(6)), asDouble(x(7)), asDouble(x(8)), asDouble(x(9)), asDouble(x(10)),
        asDouble(x(11)), asDouble(x(12)), asDouble(x(13)), asDouble(x(14)), asString(x(15))))})

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return the blockbuster dataset. It has a lot of columns (312), but not that many rows (421)
    */
  def readBlockBusterData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "blockbuster.data")
    val nullable = true
    val numTrailingNumberCols = 308

    // A whole bunch of numeric columns
    var fields: Seq[StructField] = for (i <- 1 to numTrailingNumberCols) yield {
      StructField("col" + i, DoubleType, nullable)
    }
    fields = List(List(
      StructField("Store", DoubleType, nullable),
      StructField("Sqft", DoubleType, nullable),
      StructField("City", StringType, nullable),
      StructField("State", StringType, nullable)
    ), fields).flatten

    val schema = StructType(fields)
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(
        List(Seq(asDouble(x(0)), asDouble(x(1)), asString(x(2)), asString(x(3))),
          for (i <- 4 to numTrailingNumberCols + 3) yield { asDouble(x(i)) }
        ).flatten )
      })

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return subset of 311 service call data.
    */
  def readSvcRequests40000Data(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "svcRequests40000.data")
    val nullable = true

    val schema = StructType(List(
      StructField("Unique Key", DoubleType, nullable),
      StructField("Closed Date", DoubleType, nullable),
      StructField("Agency", StringType, nullable),
      StructField("Complaint Type", StringType, nullable),
      StructField("Descriptor", StringType, nullable),
      StructField("Incident Zip", StringType, nullable),
      StructField("City", StringType, nullable),
      StructField("Landmark", StringType, nullable),
      StructField("Facility Type", StringType, nullable),
      StructField("Status", StringType, nullable),
      StructField("Borough", StringType, nullable),
      StructField("X Coordinate (State Plane)", DoubleType, nullable),
      StructField("Y Coordinate (State Plane)", DoubleType, nullable),
      StructField("Latitude", DoubleType, nullable),
      StructField("Longitude", DoubleType, nullable)
    ))

    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asDouble(x(0)), asDateDouble(x(1)), asString(x(2)), asString(x(3)), asString(x(4)),
        asString(x(5)), asString(x(6)), asString(x(7)), asString(x(8)), asString(x(9)), asString(x(10)),
        asDouble(x(11)), asDouble(x(12)), asDouble(x(13)), asDouble(x(14))))})

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return dataset with lots of rows
    */
  def readServerXData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "serverX_100000.data")
    val nullable = true

    val schema = StructType(List(
      StructField("rpm1", DoubleType, nullable),
      StructField("CPU1_TJ", DoubleType, nullable),
      StructField("CPU2_TJ", DoubleType, nullable),
      StructField("total_cfm", DoubleType, nullable),
      StructField("val1", DoubleType, nullable),
      StructField("val2", DoubleType, nullable),
      StructField("target4", StringType, nullable),
      StructField("target2", StringType, nullable)
    ))

    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asDouble(x(0)), asDouble(x(1)), asDouble(x(2)),
        asDouble(x(3)), asDouble(x(4)), asDouble(x(5)),
         asString(x(6)), asString(x(7))))})

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return dataset with lots of rows
    */
  def readServerBigXData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "serverX_10000000.data")
    val nullable = true

    val schema = StructType(List(
      StructField("targetA", StringType, nullable),
      StructField("val1", DoubleType, nullable),
      StructField("val2", DoubleType, nullable),
      StructField("val3", DoubleType, nullable),
      StructField("val4", DoubleType, nullable),
      StructField("val5", DoubleType, nullable),
      StructField("val6", DoubleType, nullable),
      StructField("targetB", StringType, nullable)
    ))

    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(asString(x(0)), asDouble(x(1)),
        asDouble(x(2)), asDouble(x(3)), asDouble(x(4)), asDouble(x(5)), asDouble(x(6)), asString(x(7))))
      })

    sqlContext.createDataFrame(rows, schema)
  }

  /** @return dataset with lots of rows
    */
  def readRedTrainData(sqlContext: SQLContext): DataFrame = {
    val data = SPARK_CTX.textFile(FILE_PREFIX + "red_train.data")
    val nullable = true

    val schema = StructType(List(
      StructField("col1", DoubleType, nullable),
      StructField("col2", DoubleType, nullable),
      StructField("col3", DoubleType, nullable),
      StructField("col4", DoubleType, nullable),
      StructField("col5", DoubleType, nullable),
      StructField("col6", DoubleType, nullable),
      StructField("col7", DoubleType, nullable),
      StructField("col8", DoubleType, nullable),
      StructField("col9", DoubleType, nullable),
      StructField("outcome", StringType, nullable)
    ))

    // ints and dates must be read as doubles
    val rows = data.map(line => line.split(",").map(elem => elem.trim))
      .map(x => {Row.fromSeq(Seq(
        asDouble(x(0)), asDouble(x(1)), asDouble(x(2)), asDouble(x(3)), asDouble(x(4)), asDouble(x(5)),
        asDouble(x(6)), asDouble(x(7)), asDouble(x(8)), asString(x(9))))
      })

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
