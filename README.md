***Minimum Description Lenght Discretizer***

This method implements Fayyad's discretizer [1] based on Minimum Description Length Principle (MDLP) in order to treat non discrete datasets from a distributed perspective. We have developed a distributed version from the original one performing some important changes.

-- Improvements:

* Support for sparse data.
* Multi-attribute processing. The whole process is carried out in a single step when the number of boundary points per attribute fits well in one partition (<= 100K boundary points per attribute).
* Support for attributes with a huge number of boundary points (> 100K boundary points per attribute). Not a normal case.

-- Example: 

  	val categoricalFeat: Option[Seq[Int]] = None
	val nBins = 25

	println("*** Discretization method: Fayyad discretizer (MDLP)")
	println("*** Number of bins: " + nBins)			

	val discretizer = MDLPDiscretizer.train(train,
			categoricalFeat, // continuous features 
			nBins) // max number of values per feature
  	discretizer
		    
	val reduced = data.map(i => LabeledPoint(i.label, discretizer.transform(i.features)))
  	reduced.first()

This software has been proved with two large real-world datasets such as:

* A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). The dataset has 32 million instances, 631 attributes, 2 classes, 98% of negative examples and occupies, when uncompressed, about 56GB of disk space.
* Epsilon dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. 400K instances and 2K attributes

We have demonstrated that our method performs 300 times faster than the sequential version for the first dataset, and also improves the accuracy for Naive Bayes with respect to the version without discretization.

Design doc: https://docs.google.com/document/d/1HOaPL_HJzTbL2tVdzbTjhr5wxVvPe9e-23S7rc2VcsY/edit?usp=sharing

**References**

[1] Fayyad, U., & Irani, K. (1993).
"Multi-interval discretization of continuous-valued attributes for classification learning."
