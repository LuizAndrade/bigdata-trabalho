import pyspark as ps
import warnings
import pandas as pd
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
    # Adicionar caminho do arquivo CSV dentro do .load
    df = (sqlContext.read
          .format('com.databricks.spark.csv')
          .options(header='true', inferschema='true')
          .load(''))
    (train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed=2000)

    # Treinamento
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="target", outputCol="label")
    pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

    pipelineFit = pipeline.fit(train_set)
    train_df = pipelineFit.transform(train_set)
    # val_df = pipelineFit.transform(val_set)
    test_df = pipelineFit.transform(test_set)
    # train_df.show(5)

    # Validação
    lr = LogisticRegression(maxIter=100)
    lrModel = lr.fit(train_df)
    # predictions = lrModel.transform(val_df)
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    # print(f"Validação: {evaluator.evaluate(predictions)}")

    # accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
    # print(f"Eficácia: {accuracy}")

    # Teste
    test_predictions = lrModel.transform(test_df)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    print(f"Validação Teste: {evaluator.evaluate(test_predictions)}")

    test_accuracy = test_predictions.filter(test_predictions.label == test_predictions.prediction).count() / float(test_set.count())
    print(f"Eficácia Teste: {test_accuracy}")

    pd_df = pd.read_csv("/home/luiz/workspace/study/python/bigdata-trabalho/trainingandtestdata/new_clean_tweets.csv")
    count_neg = len(pd_df.loc[pd_df["target"] == 0])
    count_pos = len(pd_df) - count_neg
    return {
        "count_pos": count_pos,
        "count_neg": count_neg,
        "accuracy": test_accuracy
    }
