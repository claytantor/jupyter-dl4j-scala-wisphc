# jupyter-dl4j-scala-wisphc

This project is intended to provide a comprehensive deeplearning environment
using the deeplearning4j platform. *We have noticed incosistent startup behaviours
for this conatiner, this has been resolved by restarting the Jupyter kernal and running all.* 

## Status

* Jupyter w/ Scala Kernal - https://github.com/alexarchambault/jupyter-scala
* deeplearning4j 
* nd4j-native-linux-x86 
* wisp for highcharts 

## Runing The container

```
$ docker run -d -p 8888:8888 claytantor/jupyter-dl4j-scala-wisphc:latest
```

## Notebooks

### nd4j1.ipynb
vThis workbook is intended to provide a the "MVP" of a jupyter enabled notebook for
 [deeplearning4j](https://deeplearning4j.org). Its goal is to provide a working 
 example of a DL4J network, and then to use [wisp](https://github.com/quantifind/wisp) 
 to display the results of the network classification output.
 
 ![scatter](https://raw.githubusercontent.com/claytantor/jupyter-dl4j-scala-wisphc/master/docs/img/nd4j1.png "scatter")
 
 This example is based on the classic classicication problem using 
 [iris flower data](https://en.wikipedia.org/wiki/Iris_flower_data_set). 
