Minimum Description Lenght Discretizer
========================================

This method implements Fayyad's discretizer [1] based on Minimum Description Length Principle (MDLP) in order to treat non discrete datasets from a distributed perspective. We have developed a distributed version from the original one performing some important changes.

Spark package: http://spark-packages.org/package/sramirez/spark-MDLP-discretization

## Improvements:

* Support for sparse data.
* Multi-attribute processing. The whole process is carried out in a single step when the number of boundary points per attribute fits well in one partition (<= 100K boundary points per attribute).
* Support for attributes with a huge number of boundary points (> 100K boundary points per attribute). Rare case.

This software has been proved with two large real-world datasets such as:

* A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). We have created a oversampling version of this dataset with 64 million instances, 631 attributes, 2 classes.
* kddb dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20%28bridge%20to%20algebra%29. 20M instances and almost 30M of attributes.

Design doc: https://docs.google.com/document/d/1HOaPL_HJzTbL2tVdzbTjhr5wxVvPe9e-23S7rc2VcsY/edit?usp=sharing

Publication: S. Ramírez-Gallego, S. García, H. Mouriño-Talin, D. Martínez-Rego, V. Bolón, A. Alonso-Betanzos, J.M. Benitez, F. Herrera. "Data Discretization: Taxonomy and Big Data Challenge", WIRES Data Mining and Knowledge Discovery. In press, 2015.  

## Example: 

	import org.apache.spark.mllib.feature.MDLPDiscretizer
	
  	val categoricalFeat: Option[Seq[Int]] = None
	val nBins = 25
	val maxByPart = 10000

	println("*** Discretization method: Fayyad discretizer (MDLP)")
	println("*** Number of bins: " + nBins)

	// Data must be cached in order to improve the performance
	
	val discretizer = MDLPDiscretizer.train(data, // RDD[LabeledPoint]
			categoricalFeat, // continuous features 
			nBins, // max number of thresholds by feature
			maxByPart) // max elements per partition
  	discretizer
		    
	val discrete = data.map(i => LabeledPoint(i.label, discretizer.transform(i.features)))
  	discrete.first()
  	
## Important notes:

MDLP uses *maxByPart* parameter to group boundary points by feature in order to perform an independent computation of entropy per attribute. In most of cases, a default value of 10K is enough to compute the entropy in a parallel way, thus removing iterativity implicit when we manage features with many boundary points. Log messages inform when there is a "big" feature (| boundary | > *maxByPart*) in our algorithm, which can deteriorate the performance of the algorithm. To solve this problem, it is recommended to increment the *maxByPart*'s value to 100K, or to reduce the precision of data in problems with floating-point values. 

##References

[1] Fayyad, U., & Irani, K. (1993).
"Multi-interval discretization of continuous-valued attributes for classification learning."


Please, for any comment, contribution or question refer to: https://issues.apache.org/jira/browse/SPARK-6509.
