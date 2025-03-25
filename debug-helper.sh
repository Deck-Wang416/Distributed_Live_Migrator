#!/bin/bash

set -e

echo "Current cluster status:"
kubectl get pods -l app=distributed-trainer
echo

echo "Delete the secondary node distributed-trainer-1 (simulate offline)."
kubectl delete pod distributed-trainer-1 --grace-period=0 --force
echo

echo "Wait for the master node log output RPC failed:"
sleep 10
kubectl logs distributed-trainer-0 -c trainer | tail -n 30
echo

echo "Pull up the new node distributed-trainer-2."
kubectl scale statefulset distributed-trainer --replicas=3
echo

echo "Wait for distributed-trainer-2 to start."
kubectl wait --for=condition=Ready pod distributed-trainer-2 --timeout=60s
echo

echo "See if the master node automatically retry to worker2, and the backup RPC is successful:"
kubectl logs -f distributed-trainer-0 -c trainer
