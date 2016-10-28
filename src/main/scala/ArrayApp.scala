import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by claytongraham on 10/26/16.
  */
object ArrayApp {
   def main(args: Array[String]): Unit = {
      args.foreach(println)
      println("Hello, world!")
      val arr: INDArray = Nd4j.create(Array[Float](1f, 20000000f, 40.838383f, 3f), Array[Int](2, 2))
      println(arr)
   }
}
