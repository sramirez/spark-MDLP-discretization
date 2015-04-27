import sbt._
import sbt.Keys._
import sbtsparkpackage.SparkPackagePlugin.autoImport._

object ProjectBuild extends Build {
  lazy val project = Project(
    id = "root",
    base = file("."),
    settings = Project.defaultSettings ++ Seq(
      	name := "spark-MDLP-discretization",
	version := "0.1",
	organization := "org.apache",
	scalaVersion := "2.10.4",
	spName := "apache/spark-MDLP-discretization",
	sparkVersion := "1.3.0",
	sparkComponents += "mllib",
	publishMavenStyle := true,
	licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"),
	credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")
  ))
}

