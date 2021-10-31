#!/usr/bin/env python
# coding: utf-8
import os
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit


class recommender:
    
    def __init__(self):
        os.environ['JAVA_HOME']= r"C:\Program Files\Java\jre1.8.0_311" 
        os.environ['SPARK_HOME'] = r'C:\Program Files\spark'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'jupyter'
        os.environ['PYSPARK_DRIVER_PYTHON_OPTS'] = 'notebook'
        os.environ['PYSPARK_PYTHON'] = 'python'
        
    def start_spark(self):
        conf = SparkConf() 
        self.sc = SparkContext(conf=conf).getOrCreate()
        self.spark = SparkSession.builder.getOrCreate()
    
    def read_data(self):
        self.data = self.spark.read.option("header","true").option("inferSchema","true").format("csv").load("ratings.csv")
        self.movies = self.spark.read.option('header','true').option("inferSchema","true").format("csv").load("movies.csv")
        self.data = self.data.drop('timestamp')

    def add_new_user(self,new_user_ratings,new_user_id):
        self.new_user_id = new_user_id
        self.new_user = self.spark.createDataFrame(new_user_ratings,['userId','movieId','rating'])
        self.new_user_ratings_RDD = self.sc.parallelize(new_user_ratings)
        self.total_data = self.data.union(self.new_user)
    
    def train_model(self):
        als = ALS(maxIter=5,
          regParam=0.01, 
          userCol="userId", 
          itemCol="movieId", 
          ratingCol="rating",
          coldStartStrategy="drop")
        self.model = als.fit(self.total_data)
    
    def make_recomendations(self):
        ids = self.new_user.select("movieId").rdd.flatMap(lambda x: x).collect()
        not_seen = self.total_data.filter(~self.total_data.movieId.isin(ids))
        new_df = not_seen.withColumn('userId', lit(self.new_user_id))
        self.predictions = self.model.transform(new_df)
    
    def show_recomendations(self):
        movie_preds = self.predictions.orderBy("prediction", ascending=False).dropDuplicates(['prediction']).take(10)
        recom = [movie_preds[i][1] for i in range(len(movie_preds))]
        return self.movies.filter(self.movies.movieId.isin(recom)).show()
