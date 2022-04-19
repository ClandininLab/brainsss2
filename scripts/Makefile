BASEDIR = /data/brainsss
QUEUE = normal
BASEDIR = /scratch/users/russpold/brainsss

clean:
	-rm -rf $(BASEDIR)/processed/* $(BASEDIR)/logs/* $(BASEDIR)/master_2P.xlsx

test-build-local:
	bash preprocess.sh --local -b $(BASEDIR) -v --no_require_settings --build --import_date 20220329 

test-build-local-sherlock:
	bash preprocess.sh --local -b $(BASEDIR) -v --no_require_settings --build --import_date 20220329 --import_dir /oak/stanford/groups/trc/data/Brezovec/2P_Imaging/imports 

test-proc-local:
	python preprocess.py -b $(BASEDIR) -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001 --local

test-both-local:
	python preprocess.py -b $(BASEDIR) --build --import_date 20220329 -v --no_require_settings --fictrac_qc --local

test-build-slurm:
	sbatch preprocess.sh --partition=$(QUEUE) -b $(BASEDIR) -v --no_require_settings --build --import_date 20220329

test-proc-slurm:
	sbatch preprocess.sh --partition=$(QUEUE) -b $(BASEDIR) -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001

test-both-slurm:
	sbatch preprocess.sh --partition=$(QUEUE) -b $(BASEDIR) --import_date 20220329  -v --no_require_settings --fictrac_qc --build 

test-both-slurm-sherlock:
	sbatch preprocess.sh --partition=$(QUEUE) -b $(BASEDIR) --import_date 20220329  -v --no_require_settings --fictrac_qc --build --import_dir /oak/stanford/groups/trc/data/Brezovec/2P_Imaging/imports
