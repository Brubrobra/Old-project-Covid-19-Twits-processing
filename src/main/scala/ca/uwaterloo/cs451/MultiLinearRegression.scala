/*
SentimentAnalysis takes preprocessed data from bigrams and trigrams and
performs further processing to quantify a twitter sentiment on Covid19.

spark-submit --class ca.uwaterloo.cs451.COVID19Predictor.MultiLinearRegression \\ 
    target/finalproject-1.0.jar --input data/processed --daily data/daily/Provincial_Daily_Totals.csv \\
    --output data/sentiment

Coded by: Calder Lund
*/

package ca.uwaterloo.cs451.COVID19Predictor

import org.apache.commons.io.FilenameUtils
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.rogach.scallop._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector


object MultiLinearRegression {
  val log = Logger.getLogger(getClass().getName())

  class Conf(args: Seq[String]) extends ScallopConf(args) {
    mainOptions = Seq(input, daily, output)
    val input = opt[String](descr = "processed input path", required = true)
    val daily = opt[String](descr = "daily cases input path", required = true)
    val output = opt[String](descr = "output path", required = true)
    val province = opt[String](descr = "province", required = false, default = Some("CA"))
    val predictor = opt[String](descr = "value to predict", required = false, default = Some("DailyTotals"))
    verify()
  }

  def expr(myCols: Set[String], allCols: Set[String]) = {
    allCols.toList.map(x => x match {
      case x if myCols.contains(x) => col(x)
      case _ => lit(null).as(x)
    })
  }

  def merge(df1: DataFrame, df2: DataFrame, df3: DataFrame): DataFrame = {
    val cols1 = df1.columns.toSet
    val cols2 = df2.columns.toSet
    val cols3 = df3.columns.toSet
    val total = cols1 ++ cols2 ++ cols3 // union
    df1.select(expr(cols1, total):_*)
      .unionAll(df2.select(expr(cols2, total):_*))
      .unionAll(df3.select(expr(cols3, total):_*))
  }

  def main(argv: Array[String]) {
    val args = new Conf(argv)

    log.info("Processed Input: " + args.input())
    log.info("Daily Cases Input: " + args.daily())
    log.info("Output: " + args.output())
    log.info("Province Abbreviation: " + args.province())

    val spark = SparkSession
      .builder()
      .appName("MultiLinearRegression")
      .getOrCreate() 
    spark.sparkContext.setLogLevel("ERROR")

    val terms = spark.read.option("header", "true")
      .csv(args.input() + "/terms")

    /*val bigrams = spark.read.csv(args.input() + "/bigrams")
    val trigrams = spark.read.csv(args.input() + "/trigrams")

    var grams = merge(terms, bigrams, trigrams)
      .withColumn("parsedDate", to_date(col("date"), "dd-MM-yyyy"))
      .filter(col("parsedDate").between("2020-03-21", "2020-10-13"))
    val cols = grams.columns.toArray.filter(x => x != "date" && x != "parsedDate")*/
    var grams = terms
      .withColumn("parsedDate", to_date(col("date"), "yyyy-MM-dd"))
      .filter(col("parsedDate").between("2020-03-21", "2020-10-13"))
      .orderBy("parsedDate")

    val cols = grams.columns.toArray.filter(x => x != "date" && x != "parsedDate")

    for (colName<-cols){
      grams = grams.withColumn(colName, col(colName).cast("Double"))
    }

    // Replace nan with mean
    grams = grams.na.fill(cols.zip(
      grams.select(cols.map(mean(_)): _*).first.toSeq
    ).toMap)

    val daily = spark.read.option("header", "true")
      .csv(args.daily())
      .filter(col("Abbreviation") === args.province())
      .withColumn("parsedDate", to_date(col("SummaryDate"), "yyyy/MM/dd HH:mm:ss+SS"))
      .filter(col("parsedDate").between("2020-03-21", "2020-10-13"))
      .orderBy("parsedDate")
    
    val data = grams
      .as("grams").join(daily.as("daily"), grams("parsedDate") === daily("parsedDate"))
      .withColumn("label", col(args.predictor()).cast("Double"))

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val transformed = assembler.transform(data)

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(transformed)
    // Scale features using the scaler model
    val scaledFeatures = scalerModel.transform(transformed)

    val train = scaledFeatures
      .filter(col("grams.parsedDate").lt(lit("2020-09-20")))
      .rdd.map(row => LabeledPoint(
        row.getAs[Double]("label"),   
        Vectors.fromML(row.getAs[DenseVector]("scaledFeatures"))
      ))
      
    val test = scaledFeatures
      .filter(col("grams.parsedDate").gt(lit("2020-09-19")))
    val testLabel = test.rdd.map(row => row.getAs[Double]("label"))
    val testFeatures = test.rdd.map(row => Vectors.fromML(row.getAs[DenseVector]("scaledFeatures")))

    val lrModel = LinearRegressionWithSGD.train(train, 500, 1.0)

    val predictions = lrModel.predict(testFeatures)
    val predictionAndLabel = predictions.zip(testLabel)
    predictionAndLabel.take(50).foreach(println)

    spark.stop()
  }
}
