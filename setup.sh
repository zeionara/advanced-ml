source ~/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate advanced-ml
cat requirements.txt | xargs -n 1 pip install
