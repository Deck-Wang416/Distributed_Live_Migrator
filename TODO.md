1. Loaded the pre-trained BERT-Base model using the Hugging Face Transformers library.
2. Retrieved the IMDB dataset from Kaggle for a sentiment classification task.
3. Selected 12,500 samples from the dataset: First, split the dataset into 80% training (40,000) and 20% testing (10,000). Then, randomly sampled 25% of each split, resulting in 10,000 training samples and 2,500 testing samples.
4. Enabled GPU acceleration using the Metal backend on an M2 chip.
5. Trained a sentiment classification model based on BERT-Base, using 20% of the training samples for validation and achieving approximately 81.3% accuracy on the test samples.
6. Managed large files (datasets and models sized 50–100 MB) using Git LFS for version control.
7. Implemented `save_checkpoint` and `load_checkpoint` functions to save and restore model state, optimizer state, and training progress for checkpoint management.
8. Resume from the last saved epoch after an interruption: if interrupted after saving the checkpoint for epoch 2, the training will resume from epoch 3, continuing from the saved state rather than the exact interruption point.
9. Adapted the training script for multi-CPU distributed training using `DistributedDataParallel (DDP)`, ensuring gradient synchronization across nodes.
10. Built a Docker image containing the modified training code and all required dependencies.
11. Pushed the Docker image to Docker Hub, allowing Kubernetes to pull it for deployment.
12. Deployed the distributed training system on Azure Kubernetes Service (AKS), creating the resource group `LiveMigrateResourceGroup`.
13. Configured Kubernetes deployment using YAML files, including:
   - **Headless Service**: Enables direct communication between StatefulSet pods.
   - **Persistent Volume (PV) and Persistent Volume Claim (PVC)**: Mounted dataset storage to allow shared access.
   - **StatefulSet**: Manages distributed training pods, ensuring proper rank assignment and fault recovery.
14. Implemented automated checkpointing and fault recovery:
   - If a node fails, the remaining nodes continue training.
   - The recovered node automatically loads the latest checkpoint and resumes training in sync with the cluster.
   - Only the primary process handles checkpoint saving and loading.
15. Resolved storage limitations by switching from **Azure Disk** (ReadWriteOnce) to **Azure File** (ReadWriteMany) to enable multiple nodes to share training data.
16. Primary node deletes the StatefulSet after training completion to prevent automatic restart.
17. Created and mounted Persistent Volume Claims (PVCs) for both checkpoints and trained models.
18. Added Hugging Face cache (with `emptyDir: {}`) to prevent excessive memory usage, mitigating OOMKilled and CrashLoopBackOff issues.
19. Validated data parallelism by recording checkpoints for three epochs, testing failure recovery by simulating node disconnections. Verified the well-trained model through the debug pod.
