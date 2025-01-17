/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids.tool

import scala.util.{Failure, Success, Try}
import scala.util.control.NonFatal

import com.nvidia.spark.rapids.tool.profiling.ProfileUtils.replaceDelimiter
import com.nvidia.spark.rapids.tool.qualification.QualOutputWriter
import org.apache.maven.artifact.versioning.ComparableVersion
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse

import org.apache.spark.internal.{config, Logging}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.ui.SparkPlanGraphNode

object ToolUtils extends Logging {
  // List of recommended file-encodings on the GPUs.
  val SUPPORTED_ENCODINGS = Seq("UTF-8")
  // the prefix of keys defined by the RAPIDS plugin
  val PROPS_RAPIDS_KEY_PREFIX = "spark.rapids"
  // List of keys from sparkProperties that may point to RAPIDS jars.
  // Note that we ignore "spark.yarn.secondary.jars" for now as it does not include a full path.
  val POSSIBLE_JARS_PROPERTIES = Set("spark.driver.extraClassPath",
    "spark.executor.extraClassPath",
    "spark.yarn.dist.jars",
    "spark.repl.local.jars")
  val RAPIDS_JAR_REGEX = "(.*rapids-4-spark.*jar)|(.*cudf.*jar)".r

  // Add more entries to this lookup table as necessary.
  // There is no need to list all supported versions.
  private val lookupVersions = Map(
    "311" -> new ComparableVersion("3.1.1"), // default build version
    "320" -> new ComparableVersion("3.2.0"), // introduced reusedExchange
    "331" -> new ComparableVersion("3.3.1"), // used to check for memoryOverheadFactor
    "340" -> new ComparableVersion("3.4.0")  // introduces jsonProtocolChanges
  )

  // Property to check the spark runtime version. We need this outside of test module as we
  // extend the support runtime for different platforms such as Databricks.
  lazy val sparkRuntimeVersion = {
    org.apache.spark.SPARK_VERSION
  }

  def compareVersions(verA: String, verB: String): Int = {
    Try {
      val verObjA = new ComparableVersion(verA)
      val verObjB = new ComparableVersion(verB)
      verObjA.compareTo(verObjB)
    } match {
      case Success(compRes) => compRes
      case Failure(t) =>
        logError(s"exception comparing two versions [$verA, $verB]", t)
        0
    }
  }

  def runtimeIsSparkVersion(refVersion: String): Boolean = {
    compareVersions(refVersion, sparkRuntimeVersion) == 0
  }

  def compareToSparkVersion(currVersion: String, lookupVersion: String): Int = {
    val lookupVersionObj = lookupVersions.get(lookupVersion).get
    val currVersionObj = new ComparableVersion(currVersion)
    currVersionObj.compareTo(lookupVersionObj)
  }

  def isSpark320OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "320") >= 0
  }

  def isSpark331OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "331") >= 0
  }

  def isSpark340OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "340") >= 0
  }

  def isPluginEnabled(properties: Map[String, String]): Boolean = {
    (properties.getOrElse(config.PLUGINS.key, "").contains("com.nvidia.spark.SQLPlugin")
      && properties.getOrElse("spark.rapids.sql.enabled", "true").toBoolean)
  }

  def showString(df: DataFrame, numRows: Int) = {
    df.showString(numRows, 0)
  }

  /**
   * Parses the string which contains configs in JSON format ( key : value ) pairs and
   * returns the Map of [String, String]
   * @param clusterTag  String which contains property clusterUsageTags.clusterAllTags in
   *                    JSON format
   * @return Map of ClusterTags
   */
  def parseClusterTags(clusterTag: String): Map[String, String] = {
    // clusterTags will be in this format -
    // [{"key":"Vendor","value":"Databricks"},
    // {"key":"Creator","value":"abc@company.com"},{"key":"ClusterName",
    // "value":"job-215-run-1"},{"key":"ClusterId","value":"0617-131246-dray530"},
    // {"key":"JobId","value":"215"},{"key":"RunName","value":"test73longer"},
    // {"key":"DatabricksEnvironment","value":"workerenv-7026851462233806"}]

    // case class to hold key -> value pairs
    case class ClusterTags(key: String, value: String)
    implicit val formats = DefaultFormats
    try {
      val listOfClusterTags = parse(clusterTag)
      val clusterTagsMap = listOfClusterTags.extract[List[ClusterTags]].map(
        x => x.key -> x.value).toMap
      clusterTagsMap
    } catch {
      case NonFatal(_) =>
        logWarning(s"There was an exception parsing cluster tags string: $clusterTag, skipping")
        Map.empty
    }
  }

  /**
   * Try to get the JobId from the cluster name. Parse the clusterName string which
   * looks like:
   * "spark.databricks.clusterUsageTags.clusterName":"job-557875349296715-run-4214311276"
   * and look for job-XXXXX where XXXXX represents the JobId.
   *
   * @param clusterNameString String which contains property clusterUsageTags.clusterName
   * @return Optional JobId if found
   */
  def parseClusterNameForJobId(clusterNameString: String): Option[String] = {
    var jobId: Option[String] = None
    val splitArr = clusterNameString.split("-")
    if (splitArr.contains("job")) {
      val jobIdx = splitArr.indexOf("job")
      // indexes are 0 based so adjust to compare to length
      if (splitArr.length > jobIdx + 1) {
        jobId = Some(splitArr(jobIdx + 1))
      }
    }
    jobId
  }

  // given to duration values, calculate a human readable percent
  // rounded to 2 decimal places. ie 39.12%
  def calculateDurationPercent(first: Long, total: Long): Double = {
    val firstDec = BigDecimal.decimal(first)
    val totalDec = BigDecimal.decimal(total)
    if (firstDec == 0 || totalDec == 0) {
      0.toDouble
    } else {
      val res = (firstDec / totalDec) * 100
      formatDoubleValue(res, 2)
    }
  }

  // given to duration values, calculate a human average
  // rounded to specified number of decimal places.
  def calculateAverage(first: Double, size: Long, places: Int): Double = {
    val firstDec = BigDecimal.decimal(first)
    val sizeDec = BigDecimal.decimal(size)
    if (firstDec == 0 || sizeDec == 0) {
      0.toDouble
    } else {
      val res = (firstDec / sizeDec)
      formatDoubleValue(res, places)
    }
  }

  def formatDoubleValue(bigValNum: BigDecimal, places: Int): Double = {
    bigValNum.setScale(places, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  def formatDoublePrecision(valNum: Double): String = {
    truncateDoubleToTwoDecimal(valNum).toString
  }

  def truncateDoubleToTwoDecimal(valNum: Double): Double = {
    // floor is applied after multiplying by 100. This keeps the number "as is" up-to two decimal.
    math.floor(valNum * 100) / 100
  }

  def escapeMetaCharacters(str: String): String = {
    str.replaceAll("\n", "\\\\n")
      .replaceAll("\r", "\\\\r")
      .replaceAll("\t", "\\\\t")
      .replaceAll("\f", "\\\\f")
      .replaceAll("\b", "\\\\b")
      .replaceAll("\u000B", "\\\\v")
      .replaceAll("\u0007", "\\\\a")
  }

  /**
   * Converts a sequence of elements to a single string that can be appended to a formatted text.
   * Delegates to [[com.nvidia.spark.rapids.tool.profiling.ProfileUtils.replaceDelimiter]] to
   * replace what is used as a text delimiter with something else.
   *
   * @param values the sequence of elements to join together.
   * @param separator the separator string to use.
   * @param txtDelimiter the delimiter used by the output file format (i.e., comma for CSV).
   * @return a string representation of the input sequence value. In the resulting string the string
   *         representations (w.r.t. the method toString) of all elements are separated by
   *         the string sep.
   */
  def renderTextField(values: Seq[Any], separator: String, txtDelimiter: String): String = {
    replaceDelimiter(values.mkString(separator), txtDelimiter)
  }

  def formatComplexTypes(
      values: Seq[String], fileDelimiter: String = QualOutputWriter.CSV_DELIMITER): String = {
    renderTextField(values, ";", fileDelimiter)
  }

  def formatPotentialProblems(
      values: Seq[String], fileDelimiter: String = QualOutputWriter.CSV_DELIMITER): String = {
    renderTextField(values, ":", fileDelimiter)
  }

  /**
   * Given a spark property key, this predicates checks if it is related to RAPIDS configurations.
   * Note that, "related RAPIDS properties" do not always have 'spark.rapids' prefix.
   *
   * @param sparkPropKey the spark property key
   * @return True if it is directly related to RAPIDS
   */
  def isRapidsPropKey(pKey: String): Boolean = {
    pKey.startsWith(PROPS_RAPIDS_KEY_PREFIX) || pKey.startsWith("spark.executorEnv.UCX") ||
      pKey.startsWith("spark.shuffle.manager") || pKey.equals("spark.shuffle.service.enabled")
  }

  /**
   * Checks if the given value is supported for all Ops or not.
   * @param fileEncoding the value being read from the Application configs
   * @return True if file encoding is supported
   */
  def isFileEncodingRecommended(fileEncoding: String): Boolean = {
    fileEncoding.matches("(?i)utf-?8")
  }

  /**
   * Collects the paths that points to RAPIDS jars in a map of properties.
   * @param properties the map of properties to holding the app configuration.
   * @return set of unique file paths that matches RAPIDS jars patterns.
   */
  def extractRAPIDSJarsFromProps(properties: collection.Map[String, String]): Set[String] = {
    properties.filterKeys(POSSIBLE_JARS_PROPERTIES.contains(_)).collect {
      case (_, pVal) if pVal.matches(RAPIDS_JAR_REGEX.regex) =>
        pVal.split(",").filter(_.matches(RAPIDS_JAR_REGEX.regex))
    }.flatten.toSet
  }
}

object JoinType {
  val Inner = "Inner"
  val Cross = "Cross"
  val LeftOuter = "LeftOuter"
  val RightOuter = "RightOuter"
  val FullOuter = "FullOuter"
  val LeftSemi = "LeftSemi"
  val LeftAnti = "LeftAnti"
  val ExistenceJoin = "ExistenceJoin"

  val supportedJoinTypeForBuildRight = Set(Inner, Cross, LeftOuter, LeftSemi,
    LeftAnti, FullOuter, ExistenceJoin)

  val supportedJoinTypeForBuildLeft = Set(Inner, Cross, RightOuter, FullOuter)

  val allsupportedJoinType = Set(Inner, Cross, LeftOuter, RightOuter, FullOuter, LeftSemi,
    LeftAnti, ExistenceJoin)
}

object BuildSide {
  val BuildLeft = "BuildLeft"
  val BuildRight = "BuildRight"

  val supportedBuildSides = Map(BuildLeft -> JoinType.supportedJoinTypeForBuildLeft,
    BuildRight -> JoinType.supportedJoinTypeForBuildRight)
}

object SQLMetricsStats {
  val SIZE_METRIC = "size"
  val TIMING_METRIC = "timing"
  val NS_TIMING_METRIC = "nsTiming"
  val AVERAGE_METRIC = "average"
  val SUM_METRIC = "sum"

  def hasStats(metrics : String): Boolean = {
    metrics match {
      case SIZE_METRIC | TIMING_METRIC | NS_TIMING_METRIC | AVERAGE_METRIC => true
      case _ => false
    }
  }
}

object ExecHelper {
  // regular expression to search for RDDs in node descriptions
  private val dataSetRDDRegExDescLookup = Set(
    ".*\\$Lambda\\$.*".r,
    ".*\\.apply$".r
  )
  // regular expression to search for RDDs in node names
  private val dataSetOrRDDRegExLookup = Set(
    "ExistingRDD$".r,
    "^Scan ExistingRDD.*".r,
    "SerializeFromObject$".r,
    "DeserializeToObject$".r,
    "MapPartitions$".r,
    "MapElements$".r,
    "AppendColumns$".r,
    "AppendColumnsWithObject$".r,
    "MapGroups$".r,
    "FlatMapGroupsInR$".r,
    "FlatMapGroupsInRWithArrow$".r,
    "CoGroup$".r
  )
  private val UDFRegExLookup = Set(
    ".*UDF.*".r
  )

  // we don't want to mark the *InPandas and ArrowEvalPythonExec as unsupported with UDF
  private val skipUDFCheckExecs = Seq("ArrowEvalPython", "AggregateInPandas",
    "FlatMapGroupsInPandas", "MapInPandas", "WindowInPandas")

  // Set containing execs that should be labeled as "shouldRemove"
  private val execsToBeRemoved = Set(
    "GenerateBloomFilter",   // Exclusive on AWS. Ignore it as metrics cannot be evaluated.
    "ReusedExchange",        // reusedExchange should not be added to speedups
    "ColumnarToRow"          // for now, assume everything is columnar
  )

  def isDatasetOrRDDPlan(nodeName: String, nodeDesc: String): Boolean = {
    dataSetRDDRegExDescLookup.exists(regEx => nodeDesc.matches(regEx.regex)) ||
      dataSetOrRDDRegExLookup.exists(regEx => nodeName.trim.matches(regEx.regex))
  }

  def isUDF(node: SparkPlanGraphNode): Boolean = {
    if (skipUDFCheckExecs.exists(node.name.contains(_))) {
      false
    } else {
      UDFRegExLookup.exists(regEx => node.desc.matches(regEx.regex))
    }
  }

  def shouldBeRemoved(nodeName: String): Boolean = {
    execsToBeRemoved.contains(nodeName)
  }

  ///////////////////////////////////////////
  // start definitions of execs to be ignored
  // AdaptiveSparkPlan is not a real exec. It is a wrapper for the whole plan.
  private val AdaptiveSparkPlan = "AdaptiveSparkPlan"
  // Collect Limit replacement can be slower on the GPU. Disabled by default.
  private val CollectLimit = "CollectLimit"
  // Some DDL's  and table commands which can be ignored
  private val ExecuteCreateViewCommand = "Execute CreateViewCommand"
  private val LocalTableScan = "LocalTableScan"
  private val ExecuteCreateDatabaseCommand = "Execute CreateDatabaseCommand"
  private val ExecuteDropDatabaseCommand = "Execute DropDatabaseCommand"
  private val ExecuteCreateTableAsSelectCommand = "Execute CreateTableAsSelectCommand"
  private val ExecuteCreateTableCommand = "Execute CreateTableCommand"
  private val ExecuteDropTableCommand = "Execute DropTableCommand"
  private val ExecuteCreateDataSourceTableAsSelectCommand = "Execute " +
    "CreateDataSourceTableAsSelectCommand"
  private val SetCatalogAndNamespace = "SetCatalogAndNamespace"
  private val ExecuteSetCommand = "Execute SetCommand"
  private val ResultQueryStage = "ResultQueryStage"
  private val ExecAddJarsCommand = "Execute AddJarsCommand"
  private val ExecInsertIntoHadoopFSRelationCommand = "Execute InsertIntoHadoopFsRelationCommand"
  private val ScanJDBCRelation = "Scan JDBCRelation"
  private val ScanOneRowRelation = "Scan OneRowRelation"
  private val CommandResult = "CommandResult"
  private val ExecuteAlterTableRecoverPartitionsCommand =
    "Execute AlterTableRecoverPartitionsCommand"
  private val ExecuteCreateFunctionCommand = "Execute CreateFunctionCommand"
  private val CreateHiveTableAsSelectCommand = "Execute CreateFunctionCommand"
  private val ExecuteDeleteCommand = "Execute DeleteCommand"
  private val ExecuteDescribeTableCommand = "Execute DescribeTableCommand"
  private val ExecuteRefreshTable = "Execute RefreshTable"
  private val ExecuteRepairTableCommand = "Execute RepairTableCommand"
  private val ExecuteShowPartitionsCommand = "Execute ShowPartitionsCommand"
  // DeltaLakeOperations
  private val ExecUpdateCommandEdge = "Execute UpdateCommandEdge"
  private val ExecDeleteCommandEdge = "Execute DeleteCommandEdge"
  private val ExecDescribeDeltaHistoryCommand = "Execute DescribeDeltaHistoryCommand"
  private val ExecShowPartitionsDeltaCommand = "Execute ShowPartitionsDeltaCommand"

  def getAllIgnoreExecs: Set[String] = Set(AdaptiveSparkPlan, CollectLimit,
    ExecuteCreateViewCommand, LocalTableScan, ExecuteCreateTableCommand,
    ExecuteDropTableCommand, ExecuteCreateDatabaseCommand, ExecuteDropDatabaseCommand,
    ExecuteCreateTableAsSelectCommand, ExecuteCreateDataSourceTableAsSelectCommand,
    SetCatalogAndNamespace, ExecuteSetCommand,
    ResultQueryStage,
    ExecAddJarsCommand,
    ExecInsertIntoHadoopFSRelationCommand,
    ScanJDBCRelation,
    ScanOneRowRelation,
    CommandResult,
    ExecUpdateCommandEdge,
    ExecDeleteCommandEdge,
    ExecDescribeDeltaHistoryCommand,
    ExecShowPartitionsDeltaCommand,
    ExecuteAlterTableRecoverPartitionsCommand,
    ExecuteCreateFunctionCommand,
    CreateHiveTableAsSelectCommand,
    ExecuteDeleteCommand,
    ExecuteDescribeTableCommand,
    ExecuteRefreshTable,
    ExecuteRepairTableCommand,
    ExecuteShowPartitionsCommand
  )

  def shouldIgnore(execName: String): Boolean = {
    getAllIgnoreExecs.contains(execName)
  }
}

object MlOps {
  val sparkml = "spark.ml."
  val xgBoost = "spark.XGBoost"
  val pysparkLog = "py4j.GatewayConnection.run" // pyspark eventlog contains py4j
}

object MlOpsEventLogType {
  val pyspark = "pyspark"
  val scala = "scala"
}

object SupportedMLFuncsName {
  val funcName: Map[String, String] = Map(
    "org.apache.spark.ml.clustering.KMeans.fit" -> "KMeans",
    "org.apache.spark.ml.feature.PCA.fit" -> "PCA",
    "org.apache.spark.ml.regression.LinearRegression.train" -> "LinearRegression",
    "org.apache.spark.ml.classification.RandomForestClassifier.train" -> "RandomForestClassifier",
    "org.apache.spark.ml.regression.RandomForestRegressor.train" -> "RandomForestRegressor",
    "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.train" -> "XGBoost"
  )
}

case class GpuEventLogException(message: String) extends Exception(message)

object GpuTypes {
  val A100 = "A100"
  val T4 = "T4"
  val V100 = "V100"
  val K80 = "K80"
  val P100 = "P100"
  val P4 = "P4"
  val L4 = "L4"
  val A10 = "A10"
  val A10G = "A10G"

  def getGpuMem(gpu: String): String = {
    gpu match {
      case A100 => "40960m" // A100 set default to 40GB
      case T4 => "15109m" // T4 default memory is 16G
      case V100 => "16384m"
      case K80 => "12288m"
      case P100 => "16384m"
      case P4 => "8192m"
      case L4 => "24576m"
      case A10 => "24576m"
      case A10G => "24576m"
      case _ => throw new IllegalArgumentException(s"Invalid input gpu type: $gpu")
    }
  }
}
