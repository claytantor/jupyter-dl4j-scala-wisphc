import java.text.{DecimalFormat, DecimalFormatSymbols}

import org.apache.commons.lang.StringUtils
import org.nd4j.linalg.api.complex.IComplexNDArray
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by claytongraham on 10/29/16.
  */
class NDArrayStrings {
  private var sep: String = ","
  private var padding: Int = 0
  private var decFormatNum: String = "#,###,##0"
  private var decFormatRest: String = ""
  private var decimalFormat: DecimalFormat = new DecimalFormat(decFormatNum + decFormatRest)

  def this(sep: String, precisionI: Int, decFormat: String) {
    this()
    this.decFormatNum = decFormat
    this.sep = sep
    var precision: Int = precisionI

    if (precision != 0) {
      this.decFormatRest = "."
      while (precision > 0) {
        this.decFormatRest += "0"
        precision -= 1
      }
    }
    this.decimalFormat = new DecimalFormat(decFormatNum + decFormatRest)
    val sepNgroup: DecimalFormatSymbols = DecimalFormatSymbols.getInstance
    sepNgroup.setDecimalSeparator('.')
    sepNgroup.setGroupingSeparator(',')
    decimalFormat.setDecimalFormatSymbols(sepNgroup)
  }


  /**
    * Format the given ndarray as a string
    *
    * @param arr the array to format
    * @return the formatted array
    */
  def format(arr: INDArray): String = {
    val padding: String = decimalFormat.format(arr.maxNumber)
    this.padding = padding.length
    format(arr, arr.rank)
  }

  private def format(arr: INDArray, rank: Int): String = format(arr, arr.rank, 0)

  private def format(arr: INDArray, rank: Int, offsetI: Int): String = {
    val sb: StringBuilder = new StringBuilder
    var offset: Int = offsetI
    if (arr.isScalar) {
      if (arr.isInstanceOf[IComplexNDArray]) return arr.asInstanceOf[IComplexNDArray].getComplex(0).toString
      decimalFormat.format(arr.getDouble(0))
    }
    else if (rank <= 0) ""
    else if (arr.isVector) {
      sb.append("[")
      var i: Int = 0
      while (i < arr.length) {
        {
          if (arr.isInstanceOf[IComplexNDArray]) sb.append(arr.asInstanceOf[IComplexNDArray].getComplex(i).toString)
          else sb.append(String.format("%1$" + padding + "s", decimalFormat.format(arr.getDouble(i))))
          if (i < arr.length - 1) sb.append(sep)
        }
        {
          i += 1; i - 1
        }
      }
      sb.append("]")
      sb.toString
    }
    else {
      offset = offset + 1
      sb.append("[")
      var i: Int = 0
      while (i < arr.slices) {
        {
          sb.append(format(arr.slice(i), rank - 1, offset))
          if (i != arr.slices - 1) {
            sb.append(",\n")
            sb.append(StringUtils.repeat("\n", rank - 2))
            sb.append(StringUtils.repeat(" ", offset))
          }
        }
        {
          i += 1; i - 1
        }
      }
      sb.append("]")
      sb.toString
    }
  }
}
