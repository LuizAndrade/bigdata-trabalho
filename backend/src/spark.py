import pyspark as ps
import warnings
from pyspark.sql import SparkSession

# Treinamento
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Teste
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def start_spark():
    try:
        sqlContext = SparkSession.builder.appName("SimpleApp").getOrCreate()
    except ValueError:
        warnings.warn("SparkContext already exists in this scope")
    return sqlContext


def train_data(sqlContext):
    # Update the .load path for the clean_tweet.csv
    df = (sqlContext.read
          .format('com.databricks.spark.csv')
          .options(header='true', inferschema='true')
          .load('/home/luiz/workspace/study/python/bigdata-trabalho/backend/trainingandtestdata/clean_tweet.csv'))
    (train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed=2000)

    # Treinamento
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="target", outputCol="label")
    pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

    pipelineFit = pipeline.fit(train_set)
    train_df = pipelineFit.transform(train_set)
    val_df = pipelineFit.transform(val_set)
    train_df.show(5)

    # Teste
    lr = LogisticRegression(maxIter=100)
    lrModel = lr.fit(train_df)
    predictions = lrModel.transform(val_df)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    evaluator.evaluate(predictions)
