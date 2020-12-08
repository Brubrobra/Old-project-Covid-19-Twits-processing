/*
Preprocess takes our data and processes it to a more readable form for spark.
Coded by: Calder Lund
*/

package ca.uwaterloo.cs451.COVID19Predictor

import org.apache.commons.io.FilenameUtils
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.rogach.scallop._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


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

    val baseNameOfFile = udf((longFilePath: String) => FilenameUtils.getBaseName(longFilePath).substring(0, 10))

    var termsDF = spark.read.option("header", "false")
      .csv(args.input() + "/*/*_top1000terms.csv")
      .withColumnRenamed("_c0", "term")
      .withColumnRenamed("_c1", "amount")
      .withColumn("date", baseNameOfFile(input_file_name))
      .write.option("header","true")
      .csv(args.output() + "/terms")

    var bigramsDF = spark.read.option("header", "true")
      .csv(args.input() + "/*/*_top1000bigrams.csv")
      .withColumnRenamed("_c0", "bigram")
      .withColumnRenamed("_c1", "amount")
      .withColumn("date", baseNameOfFile(input_file_name))
      .write.option("header","true")
      .csv(args.output() + "/bigrams")

    var trigramsDF = spark.read.option("header", "true")
      .csv(args.input() + "/*/*_top1000trigrams.csv")
      .withColumnRenamed("_c0", "trigram")
      .withColumnRenamed("_c1", "amount")
      .withColumn("date", baseNameOfFile(input_file_name))
      .write.option("header","true")
      .csv(args.output() + "/trigrams")
  }
}
