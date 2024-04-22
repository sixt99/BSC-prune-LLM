set -xe
podman build --arch=ppc64le -t prune_llm .
podman save prune_llm:latest -o prune_llm.tar
scp prune_llm.tar bsc03268@plogin1.bsc.es:prune_llm
ssh power "cd prune_llm && rm -f prune_llm.sif && module load singularity && singularity build prune_llm.sif docker-archive:prune_llm.tar"
