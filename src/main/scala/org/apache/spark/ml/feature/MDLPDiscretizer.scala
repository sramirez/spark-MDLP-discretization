/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.ml.attribute._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}

/**
 * Params for [[MDLPDiscretizer]] and [[DiscretizerModel]].
 */
private[feature] trait MDLPDiscretizerParams extends Params with HasInputCol with HasOutputCol with HasLabelCol {

  /**
   * Maximum number of buckets into which data points are grouped.
   * Must be >= 2. default: 2
   *
   * @group param
   */
  val maxBins = new IntParam(this, "maxBins", "Maximum number of bins" +
    "into which data points are grouped. Must be >= 2.",
    ParamValidators.gtEq(2))
  setDefault(maxBins -> 2)

  /** @group getParam */
  def getMaxBins: Int = getOrDefault(maxBins)

  /**
   * Maximum number of elements to evaluate in each partition.
   * If this parameter is bigger then the evaluation phase will be sequentially performed.
   * Should be >= 10,000, but it is allowed to go as low as 100 for testing purposes.
   * default: 10,000
   * @group param
   */
  val maxByPart = new IntParam(this, "maxByPart", "Maximum number of elements per partition" +
    "to consider in each evaluation process. Should be >= 10,000 (the default)",
    ParamValidators.gtEq(100))
  setDefault(maxByPart -> 10000)

  /** @group getParam */
  def getMaxByPart: Int = getOrDefault(maxByPart)

  val stoppingCriterion = new DoubleParam(this, "stoppingCriterion",
    "The minimum description length principal (MDLP) stopping criterion. " +
    "The default is 0 to match what was suggested in the original 1993 paper by Fayyad and Irani. " +
    "However, smaller values (like -1e-4 or -1e-2) can be set in order to " +
    "relax the constraint and produce more splits.",
     ParamValidators.ltEq(0))
  setDefault(stoppingCriterion -> 0)

  /** @group getParam */
  def getStoppingCriterion: Double = getOrDefault(stoppingCriterion)

  val minBinPercentage = new DoubleParam(this, "minBinPercentage",
    "A lower limit on the percent of total instances allowed in a single bin. " +
    "The default is 0 to be consistent with the original 1993 paper by Fayyad and Irani, " +
    "but some applications may find it useful to prevent bins with just a very small number of instances." +
    " A value of 0.1% is reasonable, but it depends somewhat on the value of maxBins.",
    ParamValidators.inRange(0, 5.0))
  setDefault(minBinPercentage -> 0)

  /** @group getParam */
  def getMinBinPercentage: Double = getOrDefault(minBinPercentage)

  /** @group getParam */
  val approximate = new BooleanParam(this, "approximate",
    "If approximative DMDLP is executed. This version is faster but non-deterministic. " +
    "The final set of cut points may be slightly affected by this option. " +
    "The default is true, faster execution is preferable in large-scale environments. ")
  setDefault(approximate -> false)

  def getApproximate: Boolean = getOrDefault(approximate)
}

/**
 * :: Experimental ::
 * MDLPDiscretizer trains a model to discretize vectors using a matrix of buckets (different values for each feature).
 */
@Experimental
class MDLPDiscretizer (override val uid: String) extends Estimator[DiscretizerModel] with MDLPDiscretizerParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("MDLPDiscretizer"))

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  def setMaxByPart(value: Int): this.type = set(maxByPart, value)

  /** @group setParam */
  def setStoppingCriterion(value: Double): this.type = set(stoppingCriterion, value)

  /** @group setParam */
  def setMinBinPercentage(value: Double): this.type = set(minBinPercentage, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setApproximate(value: Boolean): this.type = set(approximate, value)


  /**
   * Computes a [[DiscretizerModel]] that contains the cut points (splits) for each input feature.
   */
  @Since("2.1.0")
  override def fit(dataset: Dataset[_]): DiscretizerModel = {

    transformSchema(dataset.schema, logging = true)
    import dataset.sparkSession.implicits._
    val input = dataset.select($(labelCol), $(inputCol)).map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }
    input.rdd.cache() // cache the input to avoid performance warning (see issue #18)
    val discretizer = org.apache.spark.mllib.feature.MDLPDiscretizer
        .train(input, None, $(maxBins), $(maxByPart), $(stoppingCriterion), $(minBinPercentage), $(approximate))
    copyValues(new DiscretizerModel(uid, discretizer.thresholds).setParent(this))
  }

  @Since("1.6.0")
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): MDLPDiscretizer = defaultCopy(extra)
}

@Since("1.6.0")
object MDLPDiscretizer extends DefaultParamsReadable[MDLPDiscretizer] {

  @Since("1.6.0")
  override def load(path: String): MDLPDiscretizer = super.load(path)
}

/**
 * :: Experimental ::
 * Model fitted by [[MDLPDiscretizer]].
 *
 * @param splits Splits organized by feature. Each column is the list of splits for a single feature.
 */
@Experimental
class DiscretizerModel private[ml] (
    override val uid: String,
    val splits: Array[Array[Float]])
  extends Model[DiscretizerModel] with MDLPDiscretizerParams with MLWritable {

  import DiscretizerModel._

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Transform a vector to the discrete space determined by the splits.
   * NOTE: Vectors to be transformed must be the same length
   * as the source vectors given to MDLPDiscretizer.fit.
   */
  @Since("2.1.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    val newSchema = transformSchema(dataset.schema, logging = true)
    val metadata = newSchema.fields.last.metadata
    val discModel = new feature.DiscretizerModel(splits)
    val discOp = udf { x:Vector=>discModel.transform(OldVectors.fromML(x)).asML}
    dataset.withColumn($(outputCol), discOp(col($(inputCol))).as($(outputCol), metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    //validateParams()
    val buckets = splits.map(_.sliding(2).map(bucket => bucket.mkString(", ")).toArray)
    val featureAttributes: Seq[attribute.Attribute] = for(i <- splits.indices) yield new NominalAttribute(
        isOrdinal = Some(true),
        values = Some(buckets(i)))
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes.toArray)
    val outputFields = schema.fields :+ newAttributeGroup.toStructField()
    StructType(outputFields)
  }  

  override def copy(extra: ParamMap): DiscretizerModel = {
    val copied = new DiscretizerModel(uid, splits)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new DiscretizerModelWriter(this)
}


@Since("1.6.0")
object DiscretizerModel extends MLReadable[DiscretizerModel] {

  private[DiscretizerModel] class DiscretizerModelWriter(instance: DiscretizerModel) extends MLWriter {

    private case class Data(splits: Array[Array[Float]])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.splits)
      val dataPath = new Path(path, "data").toString
      sqlContext.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class DiscretizerModelReader extends MLReader[DiscretizerModel] {

    private val className = classOf[DiscretizerModel].getName

    /**
     * Loads a [[DiscretizerModel]] from data located at the input path. A model
     * can be loaded from such older data (a bi-dim matrix of splits).
     *
     * @param path path to serialized model data
     * @return a [[DiscretizerModel]]
     */
    override def load(path: String): DiscretizerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val Row(splits: Array[Array[Float]]) =
          sqlContext.read.parquet(dataPath)
            .select("splits")
            .head()
      val model = new DiscretizerModel(metadata.uid, splits)
      metadata.getAndSetParams(model)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[DiscretizerModel] = new DiscretizerModelReader

  @Since("1.6.0")
  override def load(path: String): DiscretizerModel = super.load(path)
}
