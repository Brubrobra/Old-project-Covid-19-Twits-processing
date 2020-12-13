/* 
  input files format: term, count, dates
  for each term, we get its sentiment value sentiment(term)
  then set the total sentiment = count*sentiment(term)
  then we group the results by dates 
*/

package ca.uwaterloo.cs451.COVID19Predictor

import org.apache.commons.io.FilenameUtils
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.rogach.scallop._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import java.time.format.DateTimeFormatter
import ca.uwaterloo.cs451.COVID19Predictor.SentimentAnalysisUtils._
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import scala.util.Try

object SentimentAnalysis {

  val log = Logger.getLogger(getClass().getName())
  
  class SAConf(args: Seq[String]) extends ScallopConf(args) {
    mainOptions = Seq(input, output)
    val input = opt[String](descr = "input path", required = true)
    val output = opt[String](descr = "output path", required = true)
    verify()
  }

  def main(argv: Array[String]) {
    val args = new SAConf(argv)
    
    val inputPath = args.input()
    val outputPath = args.output()
    val outputDir = new Path(outputPath)

    log.info("Input path: " + inputPath )
    log.info("Output path: " + outputPath )

    val conf = new SparkConf().setAppName("SentimentAnalysis")
    val sc = new SparkContext(conf)
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val trigramRdd = sc.textFile(inputPath)
      .map(line => line.split(","))
      .filter(tokens => tokens(1) != "counts") // text, count, date
      .map(tokens => {
        val date = tokens(2).toString
        val text = tokens(0).toString
        val count = tokens(1).toInt

        val sentiment = detectSentiment(text).toDouble * count

        (date, sentiment)
      })
      .map(p => (p._1, p._2))
      .reduceByKey(_+_)
      .sortByKey()

    trigramRdd.saveAsTextFile(args.output())

    val individualValuesRdd = sc.textFile(inputPath)
      .map(line => line.split(","))
      .filter(tokens => tokens(1) != "counts")
      .map( tokens => {
        val date = tokens(2).toString
        val text = tokens(0).toString
        val count = tokens(1).toInt

        val sentiment = detectSentiment(text).toDouble

        (date, text, count, sentiment)
      })

    individualValuesRdd.saveAsTextFile(args.output() + "/singles")
  }
}
