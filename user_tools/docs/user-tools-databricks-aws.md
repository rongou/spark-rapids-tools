# RAPIDS User Tools on Databricks AWS

This is a guide for the RAPIDS tools for Apache Spark on [Databricks AWS](https://www.databricks.com/product/aws). At the end of this guide, the user will be able to run the RAPIDS tools to analyze the clusters and the applications running on Databricks AWS.

## Assumptions

The tool currently only supports event logs stored on S3 (no DBFS paths). The remote output storage is also expected to be S3.

## Prerequisites

### 1.Databricks CLI
- Install the Databricks CLI. Follow the instructions on [Databricks CLI setup & documentation](https://docs.databricks.com/dev-tools/cli/index.html).
- Set the configuration settings and credentials of the Databricks CLI:
  - Set up authentication by following these [instructions](https://docs.databricks.com/dev-tools/cli/index.html#set-up-authentication)
  - Test the authentication setup by following these [instructions](https://docs.databricks.com/dev-tools/cli/index.html#test-your-authentication-setup)
  - Verify that the access credentials are stored in the file `~/.databrickscfg` on Unix, Linux, or macOS, or in another file defined by 
    environment variable `DATABRICKS_CONFIG_FILE`.

### 2.AWS CLI

- Install the AWS CLI version 2. Follow the instructions on [aws-cli-getting-started](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html).
- Set the configuration settings and credentials of the AWS CLI by creating `credentials` and `config` files as described in [aws-cli-configure-files](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

### 3.RAPIDS tools

- Spark event logs:
  - The RAPIDS tools can process Apache Spark CPU event logs from Spark 2.0 or higher (raw, .lz4, .lzf, .snappy, .zstd)
  - For `qualification` commands, the event logs need to be archived to an accessible S3 folder.

### 4.Install the package

- Install `spark-rapids-user-tools` with python [3.8, 3.10] using:
  - pip:  `pip install spark-rapids-user-tools`
  - wheel-file: `pip install <wheel-file>`
  - from source: `pip install -e .`
- verify the command is installed correctly by running
  ```bash
    spark_rapids_user_tools databricks_aws --help
  ```

### 5.Environment variables

Before running any command, you can set environment variables to specify configurations.
- RAPIDS variables have a naming pattern `RAPIDS_USER_TOOLS_*`:
  - `RAPIDS_USER_TOOLS_CACHE_FOLDER`: specifies the location of a local directory that the RAPIDS-cli uses to store and cache the downloaded resources. The default is `/tmp/rapids_user_tools_cache`.  Note that caching the resources locally has an impact on the total execution time of the command.
  - `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY`: specifies the location of a local directory that the RAPIDS-cli uses to generate the output. The wrapper CLI arguments override that environment variable (`--local_folder` for Qualification).
- For Databricks CLI, some environment variables can be set and picked by the RAPIDS-user tools such as: `DATABRICKS_CONFIG_FILE`, `DATABRICKS_HOST` and `DATABRICKS_TOKEN`. See the description of the variables in [Environment variables](https://docs.databricks.com/dev-tools/auth.html#environment-variables).
- For AWS CLI, some environment variables can be set and picked by the RAPIDS-user tools such as: `AWS_SHARED_CREDENTIALS_FILE`, `AWS_CONFIG_FILE`, `AWS_REGION`, `AWS_DEFAULT_REGION`, `AWS_PROFILE` and `AWS_DEFAULT_OUTPUT`. See the full list of variables in [aws-cli-configure-envvars](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).

## Qualification command

### Local deployment

```
spark_rapids_user_tools databricks_aws qualification [options]
spark_rapids_user_tools databricks_aws qualification --help
```

The local deployment runs on the local development machine. It requires:
1. Installing and configuring the Databricks and AWS CLI
2. Java 1.8+ development environment
3. Internet access to download JAR dependencies from mvn: `spark-*.jar`, `hadoop-aws-*.jar`, and `aws-java-sdk-bundle*.jar`
4. Dependencies are cached on the local disk to reduce the overhead of the download.

#### Command options

| Option                         | Description                                                                                                                                                                                                                                                                                                                                                                                                 | Default                                                                                                                                                                                                                                         | Required |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| **cpu_cluster**                | The Databricks-cluster on which the Apache Spark applications were executed. Accepted values are an Databricks-cluster name, or a valid path to the cluster properties file (json format) generated by Databricks CLI command `databricks clusters get --cluster-name`                                                                                                                                      | N/A                                                                                                                                                                                                                                             |     Y    |
| **eventlogs**                  | A comma seperated list of S3 urls pointing to event logs or S3 directory                                                                                                                                                                                                                                                                                                                                    | Reads the Spark's property `spark.eventLog.dir` defined in `cpu_cluster`. This property should be included in the output of `databricks clusters get --cluster-name`. Note that the wrapper will raise an exception if the property is not set. |     N    |
| **remote_folder**              | The S3 folder where the output of the wrapper's output is copied. If missing, the output will be available only on local disk                                                                                                                                                                                                                                                                               | N/A                                                                                                                                                                                                                                             |     N    |
| **gpu_cluster**                | The Databricks-cluster on which the Spark applications is planned to be migrated. The argument can be an Databricks-cluster or a valid path to the cluster's properties file (json format) generated by the Databricks CLI `databricks clusters get --cluster-name` command                                                                                                                                 | The wrapper maps the AWS EC2 machine instances of the original cluster into AWS EC2 instances that support GPU acceleration.                                                                                                                    |     N    |
| **local_folder**               | Local work-directory path to store the output and to be used as root directory for temporary folders/files. The final output will go into a subdirectory named `qual-${EXEC_ID}` where `exec_id` is an auto-generated unique identifier of the execution.                                                                                                                                                   | If the argument is NONE, the default value is the env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory.                                                                                                   |     N    |
| **jvm_heap_size**              | The maximum heap size of the JVM in gigabytes                                                                                                                                                                                                                                                                                                                                                               | 24                                                                                                                                                                                                                                              |     N    |
| **profile**                    | A named Databricks profile that you can specify to get the settings/credentials of the Databricks account                                                                                                                                                                                                                                                                                                   | "DEFAULT"                                                                                                                                                                                                                                       |     N    |
| **aws_profile**                | A named AWS profile that you can specify to get the settings/credentials of the AWS account                                                                                                                                                                                                                                                                                                                 | "default" if the env-variable `AWS_PROFILE` is not set                                                                                                                                                                                          |     N    |
| **tools_jar**                  | Path to a bundled jar including RAPIDS tool. The path is a local filesystem, or remote S3 url                                                                                                                                                                                                                                                                                                               | Downloads the latest rapids-tools_*.jar from mvn repo                                                                                                                                                                                           |     N    |
| **filter_apps**                | Filtering criteria of the applications listed in the final STDOUT table is one of the following (`NONE`, `SPEEDUPS`, `SAVINGS`). "`NONE`" means no filter applied. "`SPEEDUPS`" lists all the apps that are either '_Recommended_', or '_Strongly Recommended_' based on speedups. "`SAVINGS`" lists all the apps that have positive estimated GPU savings except for the apps that are '_Not Applicable_'. | `SAVINGS`                                                                                                                                                                                                                                       |     N    |
| **gpu_cluster_recommendation** | The type of GPU cluster recommendation to generate. It accepts one of the following (`CLUSTER`, `JOB`, `MATCH`). `MATCH`: keep GPU cluster same number of nodes as CPU cluster; `CLUSTER`: recommend optimal GPU cluster by cost for entire cluster. `JOB`: recommend optimal GPU cluster by cost per job                                                                                                   | `MATCH`                                                                                                                                                                                                                                         |     N    |
| **verbose**                    | True or False to enable verbosity to the wrapper script                                                                                                                                                                                                                                                                                                                                                     | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                                                                                                                                               |     N    |
| **rapids_options****           | A list of valid [Qualification tool options](../../core/docs/spark-qualification-tool.md#qualification-tool-options). Note that (`output-directory`, `platform`) flags are ignored, and that multiple "spark-property" is not supported.                                                                                                                                                                    | N/A                                                                                                                                                                                                                                             |     N    |

#### Use case scenario

A typical workflow to successfully run the `qualification` command in local mode is described as follows:

1. Store the Apache Spark event logs in S3 folder.
2. A user sets up his development machine:
   1. configures Java
   2. installs Databricks CLI and configures the profile and the credentials to make sure the
      access credentials are stored in the file `~/.databrickscfg` on Unix, Linux, or macOS,
      or in another file defined by environment variable `DATABRICKS_CONFIG_FILE`.
   3. installs AWS CLI and configures the profile and the credentials to make sure the AWS CLI
      commands can access the S3 resources `LOGS_BUCKET`.
   4. installs `spark_rapids_user_tools`
3. If the results of the wrapper need to be stored on S3, then another s3 uri is required `REMOTE_FOLDER=s3://OUT_BUCKET/`
4. User defines the Databricks-cluster on which the Spark application were running. Note that the cluster does not have to be
   active; but it has to be visible by the Databricks CLI (i.e., can run `databricks clusters get --cluster-name`).
5. The following script runs qualification by passing an S3 remote directory to store the output:

   ```
   # define the wrapper cache directory if necessary
   export RAPIDS_USER_TOOLS_CACHE_FOLDER=my_cache_folder
   export EVENTLOGS=s3://LOGS_BUCKET/eventlogs/
   export CLUSTER_NAME=my-databricks-cpu-cluster
   export REMOTE_FOLDER=s3://OUT_BUCKET/wrapper_output
   
   spark_rapids_user_tools databricks_aws qualification \
      --eventlogs $EVENTLOGS \
      --cpu_cluster $CLUSTER_NAME \
      --remote_folder $REMOTE_FOLDER
   ```
   The wrapper generates a unique-Id for each execution in the format of `qual_<YYYYmmddHHmmss>_<0x%08X>`
   The above command will generate a directory containing `qualification_summary.csv` in addition to
   the actual folder of the RAPIDS Qualification tool. The directory will be mirrored to S3 path (`REMOTE_FOLDER`).

   ```
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/qualification_summary.csv
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_qualification_output/
   ```

### Qualification output

For each app, the command output lists the following fields:

- `App ID`: An application is referenced by its application ID, '_app-id_'. When running on YARN,
  each application may have multiple attempts, but there are attempt IDs only for applications
  in cluster mode, not applications in client mode.  Applications in YARN cluster mode can be
  identified by their attempt-id.
- `App Name`: Name of the application
- `Speedup Based Recommendation`: Recommendation based on '_Estimated Speed-up Factor_'. Note that an
  application that has job or stage failures will be labeled '_Not Applicable_'
- `Savings Based Recommendation`: Recommendation based on '_Estimated GPU Savings_'.
    - '_Strongly Recommended_': An app with savings GEQ 40%
    - '_Recommended_': An app with savings between (1, 40) %
    - '_Not Recommended_': An app with no savings
    - '_Not Applicable_': An app that has job or stage failures.
- `Estimated GPU Speedup`: Speed-up factor estimated for the app. Calculated as the ratio
  between '_App Duration_' and '_Estimated GPU Duration_'.
- `Estimated GPU Duration`: Predicted runtime of the app if it was run on GPU
- `App Duration`: Wall-Clock time measured since the application starts till it is completed.
  If an app is not completed an estimated completion time would be computed.
- `Estimated GPU Savings(%)`: Percentage of cost savings of the app if it migrates to an
  accelerated cluster. It is calculated as:
  ```
  estimated_saving = 100 - ((100 * gpu_cost) / cpu_cost)
  ```

The command creates a directory with UUID that contains the following:
- Directory generated by the RAPIDS qualification tool `rapids_4_spark_qualification_output`;
- A CSV file that contains the summary of all the applications along with estimated absolute costs
- Sample directory structure: 
    ```
    .
    ├── qualification_summary.csv
    └── rapids_4_spark_qualification_output
        ├── rapids_4_spark_qualification_output.csv
        ├── rapids_4_spark_qualification_output.log
        ├── rapids_4_spark_qualification_output_execs.csv
        ├── rapids_4_spark_qualification_output_stages.csv
        └── ui
    ```