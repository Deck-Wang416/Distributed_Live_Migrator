1. Load the pre-trained BERT-Base model using the Hugging Face Transformers library.
2. Retrieve the IMDB dataset from Kaggle for a sentiment classification task.
3. Select 12,500 samples from the dataset: First, split the dataset into 80% training (40,000) and 20% testing (10,000). Then, randomly sampled 25% of each split, resulting in 10,000 training samples and 2,500 testing samples.
4. Enable GPU acceleration using the Metal backend on an M2 chip.
5. Train a sentiment classification model based on BERT-Base, using 20% of the training samples for validation and achieving approximately 81.3% accuracy on the test samples.
6. Manage large files (datasets and models sized 50–100 MB) using Git LFS for version control.
7. Implement `save_checkpoint` and `load_checkpoint` functions to save and restore model state, optimizer state, and training progress for checkpoint management.
8. Resume from the last saved epoch after an interruption: if interrupted after saving the checkpoint for epoch 2, the training will resume from epoch 3, continuing from the saved state rather than the exact interruption point.
9. Adapt the training script for multi-CPU distributed training using `DistributedDataParallel (DDP)`, ensuring gradient synchronization across nodes.
10. Build a Docker image containing the modified training code and all required dependencies.
11. Push the Docker image to Docker Hub, allowing Kubernetes to pull it for deployment.
12. Deploy the distributed training system on Azure Kubernetes Service (AKS), creating the resource group `LiveMigrateResourceGroup`.
13. Configure Kubernetes deployment using YAML files, including:
   - **Headless Service**: Enables direct communication between StatefulSet pods.
   - **Persistent Volume (PV) and Persistent Volume Claim (PVC)**: Mounted dataset storage to allow shared access.
   - **StatefulSet**: Manages distributed training pods, ensuring proper rank assignment and fault recovery.
14. Implement automated checkpointing and fault recovery:
   - If a node fails, the remaining nodes continue training.
   - The recovered node automatically loads the latest checkpoint and resumes training in sync with the cluster.
   - Only the primary process handles checkpoint saving and loading.
15. Resolve storage limitations by switching from **Azure Disk** (ReadWriteOnce) to **Azure File** (ReadWriteMany) to enable multiple nodes to share training data.
16. Primary node deletes the StatefulSet after training completion to prevent automatic restart.
17. Creat and mount Persistent Volume Claims (PVCs) for both checkpoints and trained models.
18. Add Hugging Face cache (with `emptyDir: {}`) to prevent excessive memory usage, mitigating OOMKilled and CrashLoopBackOff issues.
19. Resolve state_dict key mismatch during checkpoint loading by stripping the "module." prefix in `load_checkpoint()`.
20. Fix `model.save_pretrained()` issue by unwrapping the DDP-wrapped model before saving.
21. Replace `kubectl delete statefulset` with Kubernetes API calls since `kubectl` was not available in the container.
22. Configure RBAC by creating a ClusterRole allowing StatefulSet deletion and binding it to the default ServiceAccount.
23. Validate data parallelism by recording checkpoints for three epochs, testing failure recovery by simulating node disconnections. Verified the well-trained model through the debug pod.
24. Refactor checkpointing logic, let each worker independently saves and loads its own checkpoints. Implemented `dist.barrier()` to synchronize training steps across all nodes.
25. Configure a CI/CD pipeline using GitHub Actions, integrating Docker Hub for automated image builds and Azure Kubernetes Service (AKS) for seamless deployment.
26. Partition the BERT-Base model into two stages for pipeline parallelism: Pod A (rank 0) runs Embedding + first 6 encoder layers (FrontBert); Pod B (rank 1) runs last 6 encoder layers + Pooler + classifier head (BackBert). Implemented inter-pod communication to pass activation tensors from A to B for forward propagation.
27. Replace DDP and blocking send/recv calls with asynchronous communication using PyTorch RPC, backed by the TensorPipe protocol. Rank 0 sends intermediate activations via `rpc_async`; Rank 1 completes forward pass, computes loss, performs backward pass, and returns gradients via RPC.
28. Update fault recovery for model-parallel setup: only the primary node saves checkpoints. Upon failure of any node, the entire process group restarts and restores the model state from the last checkpoint to maintain consistency.
29. Apply lazy loading for model components on worker nodes. The model instance (`worker_model`) is only initialized upon the first `remote_forward` call and reused across subsequent calls, optimizing memory and startup time.
30. Fix RPC service shutdown sequence by ensuring `rpc.shutdown()` is called before `exit()`, releasing communication ports properly and avoiding address reuse errors on restart.
31. Resolve deadlock caused by `barrier()` call on Rank 1 blocking incoming RPCs. Moved barrier synchronization to after all RPC tasks complete to maintain responsiveness.
32. Ensure symmetric node state on failure: only the primary node executes the training loop and checkpointing, while secondary nodes synchronize via barrier and remain idle until notified, enabling clean group restart and state recovery.
33. Prevent premature RPC shutdown on secondary nodes by restructuring the training logic to run only on the primary node. All nodes now shut down simultaneously after training completes.
34. Ensure worker recovery uses the correct model state: initialized BackBert on new worker nodes using the restored checkpoint from main, avoiding reinitialization from pre-trained weights.
35. Document limitations with the Gloo backend: peer addresses are statically registered at initialization, making dynamic node replacement infeasible without full process group restart. Currently, seamless migration is not achievable due to this limitation.
