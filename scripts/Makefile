BASEDIR = /data/brainsss
BASEDIR = /scratch/users/russpold/brainsss

clean:
	-rm -rf $(BASEDIR)/processed/* $(BASEDIR)/logs/* $(BASEDIR)/master_2P.xlsx

test-build-local:
	python preprocess.py --local -b $(BASEDIR) -v --no_require_settings --build --import_date 20220329 

test-proc-local:
	python preprocess.py -b $(BASEDIR) -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001 --local

test-process-local:
	python preprocess.py -b $(BASEDIR) --build --import_date 20220329 -v --no_require_settings --fictrac_qc --local

test-proc-slurm:
	sbatch preprocess.sh -b $(BASEDIR) -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001

test-both-slurm:
	sbatch preprocess.sh -b $(BASEDIR) --import_date 20220329  -v --no_require_settings --fictrac_qc --build 
