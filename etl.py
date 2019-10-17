import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    
    """ Create a spark session or connect to existing one"""
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    """
    Process the songs data from S3 storage and create the analytical tables, songs table and artists table.
    
    This function read the data in json files from the S3 storage, transforme the data into tha analytcal tables
    (songs and artists), and write it into partitioned parquet files on S3.
    
    Args:
        spark: the spark session
        input_data: the S3 bucket to read data from
        output_data: the S3 bucket to write analytics tables to
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # defined the song data schema
    song_data_schema = R([
        Fld("artist_id", Str(), False),
        Fld("artist_latitude", Str(), True),
        Fld("artist_longitude", Str(), True),
        Fld("artist_location", Str(), True),
        Fld("artist_name", Str(), False),
        Fld("song_id", Str(), False),
        Fld("title", Str(), False),
        Fld("duration", Dbl(), False),
        Fld("year", Int(), False)
    ])
    
    # read song data file
    df = spark.read.json(song_data, schema=song_data_schema)

    # extract columns to create songs table
    songs_table = df.select(
        "song_id", 
        "title", 
        "artist_id", 
        "year", 
        "duration"
    ).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(
        output_data + "songs_table.parquet",
        mode="overwrite",
        partitionBy=["year", "artist_id"]
    )

    # extract columns to create artists table
    artists_table = df.select(
        "artist_id",
        col("artist_name").alias("name"),
        col("artist_location").alias("location"),
        col("artist_latitude").alias("latitude"),
        col("artist_longitude").alias("longitude"),
    ).distinct()
    
    # write artists table to parquet files
    artists_table.write.parquet(
        output_data + "artists_table.parquet",
        mode="overwrite"
    )


def process_log_data(spark, input_data, output_data):
    """
    Process the log data from S3 storage and create the analytical tables, users table, time tables, and songsplay table.
    
    This function read the data in json files from the S3 storage, transforme the data into tha analytcal tables
    (users, time, and songplays), and write it into partitioned parquet files on S3.
    
    Args:
        spark: the spark session
        input_data: the S3 bucket to read data from
        output_data: the S3 bucket to write analytics tables to
    """    
    
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"
    
    # define the log data schema
    log_data_schema = R([
        Fld("artist", Str(), True),
        Fld("auth", Str(), False),
        Fld("firstName", Str(), True),
        Fld("gender", Str(), True),
        Fld("itemInSession", Int(), False),
        Fld("lastName", Str(), True),
        Fld("length", Dbl(), True),
        Fld("level", Str(), False),
        Fld("location", Str(), True),
        Fld("method", Str(), False),
        Fld("page", Str(), False),
        Fld("registration", Dbl(), True),
        Fld("sessionId", Int(), False),
        Fld("song", Str(), True),
        Fld("status", Int(), False),
        Fld("ts", Dbl(), False),
        Fld("userAgent", Str(), True),
        Fld("userId", Str(), True)
    ])
    

    # read log data file
    df = spark.read.json(log_data, schema=log_data_schema)
    
    # filter by actions for song plays
    df = df.filter(col("page") == "NextSong")

    # extract columns for users table    
    users_table = (
        df
        .filter(
            (col("userID") != "") & 
            (col("userID").isNotNull())
        )
        .select(
            col('userId').alias('user_id'), 
            col('firstName').alias('first_name'), 
            col('lastName').alias('last_name'), 
            "gender", 
            "level"
        ).distinct())
    
    # write users table to parquet files
    users_table.write.parquet(
        output_data + "users_table.parquet", 
        mode="overwrite"
    )

    # create datetime column from original timestamp column
    df = df.withColumn('start_time', to_timestamp(df['ts']/1000))
    
    # extract columns to create time table
    time_table = (
        df
        .withColumn("hour", hour("start_time"))
        .withColumn("day", dayofmonth("start_time"))
        .withColumn("week", weekofyear("start_time"))
        .withColumn("month", month("start_time"))
        .withColumn("year", year("start_time"))
        .withColumn("weekday", dayofweek("start_time"))
        .select("start_time", "hour", "day", "week", "month", "year", "weekday")
        .distinct()
    ) 
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(
        output_data + "time_table.parquet",
        mode="overwrite",
        partitionBy=["year", "month"]
    )

    # read in song and artist data to use for songplays table
    song_df = spark.read.parquet(output_data + "songs_table.parquet")
    artist_df = spark.read.parquet(output_data + "artists_table.parquet")
    
    song_df = (
        song_df
        .join(artist_df, "artist_id", "full")
        .select("song_id", "title", "artist_id", "name", "duration")
    )
    
    # join the song data with log data and save it in songplays_table variable
    songplays_table = df.join(
        song_df,
        [df.song == song_df.title,
         df.artist == song_df.name,
         df.length == song_df.duration],
        "left")
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = (
        songplays_table
        .withColumn("songplay_id", monotonically_increasing_id())
        .select(
            "songplay_id", 
            "start_time", 
            col("userId").alias("user_id"), 
            "level", 
            "song_id", 
            "artist_id", 
            col("sessionId").alias("session_id"), 
            "location", 
            col("userAgent").alias("user_agent"),
        )
        .withColumn("year", year("start_time"))
        .withColumn("month", month("start_time"))
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(
        output_data + "songplays_table.parquet",
        mode="overwrite",
        partitionBy=["year", "month"]
    )


def main():
    
    """ Perform the ETL pipeline """
    
    spark = create_spark_session()
    
    # the data files distination
    input_data = "s3a://udacity-dend/"
    
    # the output analytics tables distination
    # Add the distination S3 bucket to load the data into
    output_data = "s3a://udacity-dend/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
