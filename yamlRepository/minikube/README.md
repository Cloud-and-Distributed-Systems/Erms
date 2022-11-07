**This document lists the problems that may be encountered when using minikube to deploy instances, and their solutions.**

* Issue1: Pod metrics works incorrectly, and couldn’t get cpu metrics.

  Solution: Minikube made a change to the metric reporting timing of minikube which seemed to disable metrics altogether, this is going to be reverted in the next release of minikube.
  
  In the meantime, you can append *--extra-config=kubelet.housekeeping-interval=10s* or *--disable-optimizations* to minikube start and your issue should be resolved.

- Issue2: k8s.gcr.io and gcr.io images pull backoff.

  Solution: You can transfer the images needed to docker hub, then pull the images to local host and rename them to k8s.gcr.io or gcr.io. Btw, you have to set “imagePullPolicy:IfNotPresent”. 
  
  For details, please refer to https://github.com/anjia0532/gcr.io_mirror

* Issue3: kubelet couldn’t get the local images.

  Solution: Make sure your minikube is connected to the current Docker runtime. You can use this command“*eval $(minikube docker-env)*” to bind the minikube pod container to the Docker.
