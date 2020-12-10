/*
Preprocess takes our data and processes it to a more readable form for spark.

spark-submit --class ca.uwaterloo.cs451.COVID19Predictor.Preprocess \\ 
    target/finalproject-1.0.jar --input data/twitter --output data/processed

Coded by: Calder Lund
*/

package ca.uwaterloo.cs451.COVID19Predictor

import org.apache.commons.io.FilenameUtils
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.rogach.scallop._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType


object Preprocess {
  val log = Logger.getLogger(getClass().getName())

  class Conf(args: Seq[String]) extends ScallopConf(args) {
    mainOptions = Seq(input, output)
    val input = opt[String](descr = "twitter input path", required = true)
    val output = opt[String](descr = "output path", required = true)
    verify()
  }

  def main(argv: Array[String]) {
    val args = new Conf(argv)

    log.info("Twitter Input: " + args.input())
    log.info("Output: " + args.output())

    val spark = SparkSession
      .builder()
      .appName("Preprocess")
      .getOrCreate()

    spark.conf.set("spark.sql.pivotMaxValues", 20000)

    val baseNameOfFile = udf((longFilePath: String) => FilenameUtils.getBaseName(longFilePath).substring(0, 10))

    var termsDF = spark.read.option("header", "false")
      .csv(args.input() + "/*/*_top1000terms.csv")
      .withColumnRenamed("_c0", "gram")
      .withColumnRenamed("_c1", "counts")
      .withColumn("counts", col("counts").cast(IntegerType))
      .withColumn("date", baseNameOfFile(input_file_name))
      .filter(col("gram").isin("covid19", "coronavirus", "fake", "hoax", "trump", "schools", "canada", 
                               "distancing", "rules", "masks", "deaths", "cases", "rise", "rising", 
                               "covid", "corona", "virus", "epidemic", "business", "economy", "closing",
                               "reopening", "reopen", "close", "regulations", "strict"))
      .groupBy("date")
      .pivot("gram")
      .max("counts")
      .write.option("header", "true")
      .csv(args.output() + "/terms")

    val bigramsDF = spark.read.option("header", "true")
      .csv(args.input() + "/*/*_top1000bigrams.csv")
      .withColumn("counts", col("counts").cast(IntegerType))
      .withColumn("date", baseNameOfFile(input_file_name))
      .filter(col("gram").contains("covid") || col("gram").contains("corona") || col("gram").contains("fake") || 
              col("gram").contains("hoax") || col("gram").contains("distancing") || col("gram").contains("virus"))
      .groupBy("date")
      .pivot("gram")
      .max("counts")
      .write.option("header", "true")
      .csv(args.output() + "/bigrams")

    val trigramsDF = spark.read.option("header", "true")
      .csv(args.input() + "/*/*_top1000trigrams.csv")
      .withColumn("counts", col("counts").cast(IntegerType))
      .withColumn("date", baseNameOfFile(input_file_name))
      .filter(col("gram").contains("covid") || col("gram").contains("corona") || col("gram").contains("fake") || 
              col("gram").contains("hoax") || col("gram").contains("distancing") || col("gram").contains("virus"))
      .groupBy("date")
      .pivot("gram")
      .max("counts")
      .write.option("header", "true")
      .csv(args.output() + "/trigrams")
  }
}
