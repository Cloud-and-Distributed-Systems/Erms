# Artifact Evaluation (AE) for Erms

## Overview

Erms is an efficient resource management framework that is mainly designed for shared microservice environments with SLA guarantees. It includes three modules, offline profiling, online scaling, and online scheduling as shown below. We evaluate Erms using DeathStarBench, including three applications, Social Network, Media Services and Hotel Reservation. Each application has one or more services; each service has multiple microservices.

![Clusterarchitecture](./Erms.png)

## Environment

We deploy Erms on top of a Kubernetes (k8s) cluster and use Jaeger and Prometheus to collect application-level and OS-level metrics, respectively.

## Tips for users

* Before executing a script, please check the configuration and help messages of the script:
  * Configuration files can be found in `scripts/AE/configs/${module_name}.yaml`

  * Help messages can be printed by `bash scripts/AE/${target_script}.sh -h`

* We provide the estimated execution time for each script if necessary. Users could organize their schedule accordingly.
* Since some scripts take a long time, please build a screen: `screen -S session_name` , to maintain a long-live ssh connection.
* If it fails when running the script, users **SHOULD** delete the result directory if exist and run the script again.
* If the status of a worker node becomes NoReady, users could reboot the node to recover the Kubernetes cluster.

## Prepare for AE

We assume that you already have a kubernetes cluster that meets the following requirements.
* there are at least 4 nodes in the cluster, a master node and 3 slave nodes.
* each slave node has at least 4 CPUs and 8 GB RAM.
* the cluster contains Prometheus.
* copy and unzip `additionalFiles/*.zip` under `/root` on all slave nodes.
In addition, you should have some knowledge of [DeathStarBench](https://github.com/delimitrou/DeathStarBench/tree/master/mediaMicroservices) and understand the pre-requisites requirement to deploy it.

In order to test Erms, you firstly need to modify the configuration in `configs/*-global.yaml` according to your environment. Usually, the most important configurations are:
* `node_for_test`: This field list the node that you want to use to deploy stateless deployments.
* `node_for_infra`: This field list the node that you want to use to deploy stateful deployments.

You **HAVE TO** modify these two fields to let the test starts. You can also change other fields in the configuration files. You can check yaml files that started by `_example` to see each field's usage.

After that, you can use `main.py` to initialize the application, to let the program knows that which app you want to init, you can set the environment variable `ERMS_APP`, its possible values are: `social`, `hotel` and `media`.

You also need to update AE configurations in `AE/scripts/utils.py`, `CONFIG` variable. It's similar to those yaml files mentioned above.

## Functional Evaluation for Erms

In this part, users can evaluate the function of Erms' each module separately. For the detail of each module, users could refer to Section 3 in the paper.

### General Script Arguments

Most of the scripts support the following arguments, for more details of each script, please use `-h` to check the manual.
* `-h` or `--help`: List usage, explanation and arguments of the script.
* `-p` or `--profiling-data`: Specify which profiling data will be used in the script. If not specified, the script will use data stored in `AE/data_{app}/` folder by default.
* `-d` or `--directory`: Specify the directory of output data/figures.


### Offline Profiling

Erms use pair-wised linear functions to profile microservices performance. The profiling process takes more than two days (if using default settings, each service will cost about 9 hours). Thus, we provide collected traces. Users can optionally run the scripts to profile each microservice under varying amounts of interference and different workloads.
  
- (Optional) Profile each application. The profiling process takes about a week for all applications, so we recommend that users can skip the offline profiling process and use our profiling data. Users can find our profiling data in `AE/data/data_{app}/`.

  > ./AE/scripts/profiling-testing.sh

  ![ProfilingTesting](./doc/img/profiling-testing.png)

  > ./AE/scripts/profiling-fitting.sh

  ![ProfilingFitting](./doc/img/profiling-fitting.png)


### Online Scaling

Online Scaling determines how many containers are allocated to each microservices, which includes three parts, i.e., dependency merge, latency target computation and priority-based scheduling. Users can evaluate the end-to-end Online Scaling module or evaluate three parts separately. The output will be printed on the terminal.

* Dependency merge:

  > ./AE/scripts/dependency-merge.sh

  ![DependencyMerge](doc/img/scaling-dependency-merge.png)

- Latency target computation:

  > ./AE/scripts/latency-target-computation.sh

  ![LatencyTargetComputation](doc/img/scaling-latency-target-computation.png)
- Priority scheduling:

  > ./AE/scripts/priority-scheduling.sh

  ![PriorityScheduling](doc/img/scaling-priority-scheduling.png)
  

### Dynamic Provisioning (Online Scheduling)

With the results of online scaling (i.e., the number of allocated containers for each microservices), the dynamic provisioning module generates schedule policies that assign the allocated containers to different nodes to balance the interference across the cluster.

Run

> ./AE/scripts/dynamic-provisioning.sh

and check the printed result on the terminal.

  ![DynamicProvisioning](doc/img/scheduling.png)

## Reproducible Evaluation for Erms

In this part, users could reproduce the experimental results in the paper. We repeated each evaluation five times and adopted the median of latency to mitigate the variance in latency.

Please note that some scripts may print error messages similart to the following:

```
Error from server (NotFound): error when deleting "tmp/scheduledAPP/cast-info-service_Deployment.yaml": deployments.apps "cast-info-service" not found
```

To initialize the running environment, we will kill the application pods first before running the applications. If there are no such pods, Kubernetes system will throw an error. This error can be ignored and the script will keep on going.

---

### Microservice Profiling Accuracy

### Resource Efficiency and Performance

**Evaluate Erms under static workload:**

```bash
# Generate theoretical result
bash ./AE/scripts/theoretical-resource.sh
# Generate experiment result
bash ./AE/scripts/static-workload.sh
```

*Notes: This process may take about 6 hours.*

**Evaluate Erms under dynamic workload:**

```bash
bash ./AE/scripts/dynamic-workload.sh
```

*Notes: This process may take about 20 hours.*

---

### Evaluation of Different Modules

**Benefit of Priority Scheduling:**

```bash
bash ./AE/scripts/benefit-priority-scheduling.sh
```

**Benefit of Interference-based Scheduling:**

```bash
bash ./AE/scripts/interference-scheduling.sh
```

*Notes: This process may take about 1 day.*

---

## How to reuse Erms beyond the paper

In this part, we introduce some tips about how to reuse Erms.

1. The project is separated into different modules. Users could modify each individual module to build their own systems. For example, users can modify latency target computation to design a new algorithm for resource allocation.
2. We use Yaml, which is easier for people to read, to configure the argument for Erms. Users could revise the Yaml file instead of the code to run Erms easily under different configurations.
