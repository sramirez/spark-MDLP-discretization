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
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType, DoubleType}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.attribute._

/**
 * Params for [[MDLPDiscretizer]] and [[DiscretizerModel]].
 */
private[feature] trait MDLPDiscretizerParams extends Params with HasInputCol with HasOutputCol with HasLabelCol {

  /**
   * Maximum number of buckets into which data points are grouped. 
   * Must be >= 2. default: 2
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
   * Must be >= 10,000.
   * default: 10,000
   * @group param
   */
  val maxByPart = new IntParam(this, "maxByPart", "Maximum number of elements per partition" +
    "to considere in each evaluation process. Must be >= 10,000.",
    ParamValidators.gtEq(10000))
  setDefault(maxByPart -> 10000)

  /** @group getParam */
  def getMaxByPart: Int = getOrDefault(maxByPart)

}

/**
 * :: Experimental ::
 * MDLPDiscretizer trains a model to discretize vectors using a matrix of buckets (different values for aech feature).
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
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  
  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /**
   * Computes a [[DiscretizerModel]] that contains the cut points (splits) for each input feature.
   */
  override def fit(dataset: DataFrame): DiscretizerModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(labelCol), $(inputCol)).map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }
    val discretizer = feature.MDLPDiscretizer.train(input, None, $(maxBins), $(maxByPart))
    copyValues(new DiscretizerModel(uid, discretizer.thresholds).setParent(this))
  }
  
  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
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
   * as the source vectors given to [[MDLPDiscretizer.fit()]].
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val discModel = new feature.DiscretizerModel(splits)
    val discOp = udf { discModel.transform _ }
    dataset.withColumn($(outputCol), discOp(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    val buckets = splits.map(_.sliding(2).map(bucket => bucket.mkString(", ")).toArray)
    val featureAttributes: Seq[attribute.Attribute] = for(i <- 0 until splits.length) yield new NominalAttribute(
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
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[DiscretizerModel] = new DiscretizerModelReader

  @Since("1.6.0")
  override def load(path: String): DiscretizerModel = super.load(path)
}
