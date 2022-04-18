BASEDIR = /data/brainsss
clean:
	-rm -rf $(BASEDIR)/processed/* $(BASEDIR)/logs/* $(BASEDIR)/master_2P.xlsx
test-proc:
	python preprocess.py -b /scratch/users/russpold/brainsss -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001
test-slurm:
	sbatch preprocess.sh -b $(BASEDIR) -v --no_require_settings --fictrac_qc --process $(BASEDIR)/processed/fly_001
test-both-slurm:
	sbatch preprocess.sh -b $(BASEDIR) --import_date 20220329  -v --no_require_settings --fictrac_qc --build 
