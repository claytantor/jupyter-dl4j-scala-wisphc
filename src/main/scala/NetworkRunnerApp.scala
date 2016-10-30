import org.slf4j.Logger
import org.slf4j.LoggerFactory


/**
  * Created by claytongraham on 10/26/16.
  */
object NetworkRunnerApp {

   private val log: Logger = LoggerFactory.getLogger("ArrayApp")

   def main(args: Array[String]): Unit = {
     val network = new DeepLearning4JMultiLayerNetwork(args{0});
     network.execute()
   }
}
