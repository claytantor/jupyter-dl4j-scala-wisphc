import java.io.{File, IOException}

import org.apache.commons.io.FileUtils
import java.net.URL

import org.nd4j.linalg.factory.Nd4j
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import java.util

import com.fasterxml.jackson.core.JsonParseException
import com.fasterxml.jackson.databind.ObjectMapper
import com.google.common.collect.{HashBasedTable, Table}
import org.datavec.common.data.NDArrayWritable
import com.quantifind.charts.highcharts.SeriesType
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts.Highchart
import com.quantifind.charts.highcharts.Histogram
/**
  * Created by claytongraham on 10/26/16.
  */
class DeepLearning4JMultiLayerNetwork(val filePath: String) {

  var dataFile: String = filePath

  var series: Highchart = histogram(Seq(1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 7), 7)

  def execute() {
    System.out.println("Hello, deeplearning4j!")

    //verify that nd4j is working
    val arr: INDArray = Nd4j.create(Array[Float](1f, 20000000f, 40.838383f, 3f), Array[Int](2, 2))
    System.out.println(arr.toString)

    //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
    val numLinesToSkip: Int = 0
    val delimiter: String = ","
    val recordReader: RecordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new File(dataFile)))

    //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
    val labelIndex: Int = 4 //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
    val numClasses: Int = 3 //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
    val batchSize: Int = 150 //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

    val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)

    val allData: DataSet = iterator.next
    allData.shuffle()
    val testAndTrain: SplitTestAndTrain = allData.splitTestAndTrain(0.65) //Use 65% of data for training
    val trainingData: DataSet = testAndTrain.getTrain
    val testData: DataSet = testAndTrain.getTest

    //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
    val normalizer: DataNormalization = new NormalizerStandardize
    normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
    normalizer.transform(trainingData) //Apply normalization to the training data
    normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set
    val numInputs: Int = 4
    val outputNum: Int = 3
    val iterations: Int = 1000
    val seed: Long = 6

    System.out.println("Build model....")
    val conf: MultiLayerConfiguration =
      new NeuralNetConfiguration.Builder().seed(seed)
        .iterations(iterations).activation("tanh")
        .weightInit(WeightInit.XAVIER).learningRate(0.1)
        .regularization(true).l2(1e-4).list
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build)
        .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build)
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation("softmax")
          .nIn(3).nOut(outputNum).build).backprop(true).pretrain(false).build

    //run the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    model.fit(trainingData)

    //evaluate the model on the test set
    val eval: Evaluation = new Evaluation(3)
    val output: INDArray = model.output(testData.getFeatureMatrix)
    eval.eval(testData.getLabels, output)
    System.out.println(eval.stats)
    System.out.println(eval)


    series = histogram(Seq(8, 14, 3, 23), 4)
  }


  @throws[IOException]
  def makeRowsFromNDArray(source: INDArray, precision: Int): util.List[_] = {
    val mapper: ObjectMapper = new ObjectMapper
    val serializedData: String = new NDArrayStrings(",", precision, "######0").format(source)
    try {
      val rows: util.List[_] = mapper.readValue(serializedData.getBytes, classOf[util.List[_]]).asInstanceOf[util.List[_]]
      rows
    }
    catch {
      case e: JsonParseException => {
        e.printStackTrace()
        return null
      }
    }
  }

  @throws[IOException]
  def makeTableFromArray(source: INDArray, precision: Int): Table[Integer, Integer, Double] = {
    val table: Table[Integer, Integer, Double] = HashBasedTable.create[Integer, Integer, Double]
    val rows: util.List[_] = makeRowsFromNDArray(source, precision)
    var i: Int = 0
    while (i < rows.size) {
      {
        val row: util.List[Double] = rows.get(i).asInstanceOf[util.List[Double]]
        var j: Int = 0
        while (j < row.size) {
          {
            table.put(i, j, row.get(j))
          }
          {
            j += 1; j - 1
          }
        }
      }
      {
        i += 1; i - 1
      }
    }
    table
  }

}