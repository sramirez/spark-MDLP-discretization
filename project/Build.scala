import sbt._
import sbt.Keys._
import sbtsparkpackage.SparkPackagePlugin.autoImport._

object ProjectBuild extends Build {
  lazy val project = Project(
    id = "root",
    base = file("."),
    settings = Project.defaultSettings ++ Seq(
      	name := "spark-MDLP-discretization",
	version := "0.2-SNAPSHOT",
	organization := "org.apache.spark",
	scalaVersion := "2.11.6",
	spName := "apache/spark-MDLP-discretization",
	sparkVersion := "1.6.2",
	sparkComponents += "mllib",
	publishMavenStyle := true,
	licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"),
	credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),

	libraryDependencies ++= Seq(
		"joda-time" % "joda-time" % "2.9.4",
		// dependencies for unit tests
		"org.scalatest" %% "scalatest" % "2.2.4" % "test",
		"junit" % "junit" % "4.12" % "test",
		"org.apache.commons" % "commons-lang3" % "3.4" % "test"
		)
	))

}

