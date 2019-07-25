# stable-baselines-wiener
Running Stable Baselines on Wiener using Singularity.

## Preface
Wiener currently uses Singularity 2.6.0 and doesn't like it if you try to use a container that was built with a newer version (I tried building a container on my local machine using Singularity 3.2 and then running the container on Wiener, it complained about an unknown file type).
It's a P.I.T.A., but to get around it you can just build the container on your local machine with the same version used on wiener - then it's happy.
So...

## Step 1: Install Singularity 2.6 on your local machine
See: https://sylabs.io/guides/2.6/user-guide/ for instructions

## Step 2: Build the stable-baselines container on your local machine using my recipe file
`sudo singularity build stable_baselines.simg stable_baselines.def`\
It will take a while...

## Step 3: Copy the container recipe and test python file to wiener
Self explanatory ;)

## Step 4: Log onto wiener and load requried modules
`module load cuda`\
`module load singularity`

## Step 5: Get an interactive slurm allocation on a GPU node
See http://www2.rcc.uq.edu.au/hpc/guides/index.html?secure/Wiener_userguide.html#Singularity for helpful info pertaining to remaining sections
`salloc -N 1 -p gpu --gres=gpu:tesla:1 --mem=10G`

## Step 6: Interactively run a shell inside the container on the node (with --nv passthrough enabled for cuda)
`srun --pty singularity shell --nv stable_baselines.simg`

## Step 7: Run the dqn test python file inside the container (trains a dqn on breakout for a few timesteps)
`python dqn.py`

## Step 8: Profit
You should see some info about tensorflow device allocation being spat out (should allocate a tesla V-100 device), then see some info about total reward printed out.
Ignore any warnings about MPI or tf deprecations.

