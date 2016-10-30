

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.util
import com.quantifind.charts.Highcharts._
import collection.mutable._
import scala.collection.JavaConversions._
import scala.collection.mutable

/**
  * Created by claytongraham on 10/26/16.
  */
object NetworkRunnerApp {

   private val log: Logger = LoggerFactory.getLogger("ArrayApp")

   def main(args: Array[String]): Unit = {
     val network = new DeepLearning4JMultiLayerNetwork(args{0});
     var listmax: List[Int] = List()
     var listrows: List[Int] = List()
     network.execute()
     val setrows : mutable.Set[Integer] = asScalaSet(network.networkOutput.rowKeySet())
     println(setrows)
     for(rowKey <- setrows) {
       var jrow: util.Map[Integer, Double] = network.networkOutput.row(rowKey)
       var rowMap: Map[Integer, Double] = mapAsScalaMap(jrow)
       var maxValColumn: Int = 0
       var maxVal: Double = 0.0
       listrows :::= List(rowKey)
       for(colKey <- rowMap.keysIterator){
          var dval = rowMap.get(colKey).get
          if(dval>maxVal){
            maxVal = dval
            maxValColumn = colKey
          }
       }

       listmax :::= List(maxValColumn)

     }

     val chart = scatter(listrows,listmax)

   }



}
