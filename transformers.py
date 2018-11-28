from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml import Pipeline, Transformer


class ChangeValue(Transformer, HasInputCol, HasOutputCol):

  @keyword_only
  def __init__(self, column=None, value_initial=None, value_change=None):
    super(ChangeValue, self).__init__()
    self.column = Param(self, "column", "")
    self.value_initial = Param(self, "value_initial", "")
    self.value_change = Param(self, "value_change", "")
    self._setDefault(column=column)
    self._setDefault(value_initial=value_initial)
    self._setDefault(value_change=value_change)

  def getColumn(self):
    return self.getOrDefault(self.column)

  def getInitialValue(self):
    return self.getOrDefault(self.value_initial)

  def getChangeValue(self):
    return self.getOrDefault(self.value_change)


  def _transform(self, df):
    inputColumn = self.getColumn()
    value_initial = self.getInitialValue()
    value_change = self.getChangeValue()

    df = df.withColumn(inputColumn, (f.when(df[inputColumn] == value_initial, value_change)).otherwise(df[inputColumn]))
    return df

class Divider(Transformer, HasInputCol, HasOutputCol):

  @keyword_only
  def __init__(self, inColumn1=None, inColumn2=None, outColumn=None):
    super(Divider, self).__init__()
    self.inColumn1 = Param(self, "inColumn1", "")
    self.inColumn2 = Param(self, "inColumn2", "")
    self.outColumn = Param(self, "outColumn", "")
    self._setDefault(inColumn1=inColumn1)
    self._setDefault(inColumn2=inColumn2)
    self._setDefault(outColumn=outColumn)

  def getInColumn1(self):
    return self.getOrDefault(self.inColumn1)

  def getInColumn2(self):
    return self.getOrDefault(self.inColumn2)

  def getOutColumn(self):
    return self.getOrDefault(self.outColumn)


  def _transform(self, df):
    inColumn1 = self.getInColumn1()
    inColumn2 = self.getInColumn2()
    outColumn = self.getOutColumn()

    df = df.withColumn(outColumn, df[inColumn1]/df[inColumn2])
    return df
