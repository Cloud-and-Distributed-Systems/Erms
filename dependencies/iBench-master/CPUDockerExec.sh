docker exec -it `docker ps | grep ibench | awk '{print $1}'` /bin/bash
