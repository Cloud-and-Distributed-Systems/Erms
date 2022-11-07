kubectl apply -f k8s-yaml/db/
count=0
while [ $count -ne 0 ];
do
   let count=`kubectl get pod -n media-microsvc|grep 0\/1|wc -l`
   echo $count
   sleep 2
done
echo "DB Ready"
kubectl apply -f k8s-yaml/service/
count=0
while [ $count -ne 0 ];
do
   let count=`kubectl get pod -n media-microsvc|grep 0\/1|wc -l`
   echo $count
   sleep 2
done
echo "service ready"
sleep 60
python3 scripts/write_movie_info.py -c datasets/tmdb/casts.json -m datasets/tmdb/movies.json  --server_address http://localhost:30092
bash scripts/register_users.sh 172.169.8.178:30412
bash scripts/register_movies.sh 172.169.8.178:30412