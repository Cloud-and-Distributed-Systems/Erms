kubectl apply -Rf yamlRepository/socialNetwork/db
count=0
while [ $count -ne 1 ];
do
   let count=`kubectl get pod -n social-network|grep 0\/1|wc -l`
   echo $count
   sleep 2
done
echo "DB Ready"
kubectl apply -Rf yamlRepository/socialNetwork/test
count=0
while [ $count -ne 1 ];
do
   let count=`kubectl get pod -n social-network|grep 0\/1|wc -l`
   echo $count
   sleep 2
done
echo "service ready"
python3 scripts/init_social_graph.py