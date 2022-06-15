## New proposed file organization scheme

This BIDS-inspired layout was designed to make understanding of the data structure easier.  General principles:

- key-value organization, with different pairs separated by underscores and key-value pairs separated by dashes
- The following keywords are used to specify particular metadata tags:
    - `ind`: refers to the indicator used for the particular image (e.g. `ind-tdTomato` or `ind-GCaMP7`)
    - `acq`: refers to the intention of the scan as either a functional (`acq-func`) or anatomical (`acq-anat`) scan
    - `scan`: refers to the scan number when there are multiple scans of the same acquisition type
- the inheritance principle from BIDS is used, such that more specific labels will inherit information from less specific labels.  in particular, information that is relevant to both indidicators will not include an indicator in the filename.
- one main way in which this layout violates the BIDS format is that the raw and derivative data live within the same directory tree. This was done to retain similarity with the previous format.

### Example layout:

    fly-001/        
        func/
            raw/   # formerly labeled "imaging", contains raw imaging data
                fly-001_acq-func_ind-tdTomato_scan-001.nii
                fly-001_acq-func_ind-GCaMP7_scan-001.nii
                fly-001_acq-func_scan-001.json (contains info previously in scan.json)
                fly-001_acq-func_scan-001.xml (contains info previously in functional.xml)
                fly-001_acq-func_scan-001_timestamps.h5 (contains info previously in timestamps.h5)
                # ?????? what about voltage_output.xml and TSeries-..._VoltageRecording_001.xml?
            
            QC/ # directory for QC outputs
                fly-001_acq-func_ind-GCaMP7_scan-001_bleaching.png
                fly-001_acq-func_ind-GCaMP7_scan-001_fictrac-hist.png
                fly-001_acq-func_ind-GCaMP7_scan-001_fictrac-trace.png
                fly-001_acq-func_ind-GCaMP7_scan-001_stimtrigturning.png

            fictrac/ # directory for fictrac outputs
                fly-001_acq-func_scan-001_fictrac.dat
                fly-001_acq-func_scan-001_fictrac.log
                fly-001_acq-func_scan-001_fictrac.xml

            visual/
                fly-001_acq-func_scan-001_visual.h5
                fly-001_acq-func_scan-001_photodiode.h5
                fly-001_acq-func_scan-001_photodiode.csv
                fly-001_acq-func_scan-001_stimulus-metadata.pkl

            logs/ # home for all logs generated from func processing

#### the following directories are generated through preprocessing

            preproc/  # home to preprocessed imaging files
                fly-001_acq-func_ind-GCaMP7_scan-001_moco.h5
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-mean.nii
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-mean_mask-ants.nii
                fly-001_acq-func_ind-GCaMP7_scan-001_moco_smoooth-2.0um.h5

            moco/ # directory for motion correction outputs
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-settings.json
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-parameters.csv
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-framewisedisplacement.csv
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-parameters.png
                fly-001_acq-func_ind-GCaMP7_scan-001_moco-extendedparameters.csv

            regression/ # directory for regression outputs
                model-000_label-confound/  # by convention the confound model is numbered zero
                    fly-001_model-000_scan-001_desmtx.csv
                    fly-001_model-000_scan-001_stat-rsquared.nii
                    fly-001_model-000_scan-001_stat-rsquared.png
                model-001_label-dRotLabYZ/
                    fly-001_model-001_scan-001_desmtx.csv
                    fly-001_model-001_scan-001_stat-R2.nii
                    fly-001_model-001_scan-001_stat-deltaR2.nii
                    fly-001_model-001_scan-001_stat-R2.png
                    fly-001_model-001_scan-001_reg-dRotLabY_stat-beta.nii
                    fly-001_model-001_scan-001_reg-dRotLabY_stat-tstat.nii
                    fly-001_model-001_scan-001_reg-dRotLabY_stat-negP.nii
                    fly-001_model-001_scan-001_reg-dRotLabY_stat-negP_corr-FDR.nii
                    ## expressing Z- conflicts with the key-value convention but we can parse it
                    fly-001_model-001_scan-001_reg-dRotLabZ-_stat-beta.nii
                    fly-001_model-001_scan-001_reg-dRotLabZ-_stat-tstat.nii
                    fly-001_model-001_scan-001_reg-dRotLabZ-_stat-negP.nii
                    fly-001_model-001_scan-001_reg-dRotLabZ-_stat-negP_corr-FDR.nii

            clustering/ # directory for "supervoxels" outputs

                    fly-001_acq-func_ind-GCaMP7_scan-001_cluster-labels.npy  # consider moving to h5 instead of npy
                    fly-001_acq-func_ind-GCaMP7_scan-001_cluster-labels.nii
                    fly-001_acq-func_ind-GCaMP7_scan-001_cluster-means.npy
                    fly-001_acq-func_ind-GCaMP7_scan-001_cluster-info.json # contain provenance info

### There are also files generated at the dataset level

        atlasreg/ # directory for atlas registration outputs
            transforms/ # use the fixed/moving convention from ANTs - assume that all refer to tdTomato indicator
            # ??? *do we need to accomodate possible multiple anat scans as well*?
                fly-001_fixed-atlas_moving-anat_0GenericAffine.mat
                fly-001_fixed-atlas_moving-anat_InverseWarp.nii.gz
                fly-001_fixed-atlas_moving-anat_Warp.nii.gz
                fly-001_fixed-anat_moving-func_scan-001_0GenericAffine.mat
                fly-001_fixed-anat_moving-func_scan-001_InverseWarp.nii.gz
                fly-001_fixed-anat_moving-func_scan-001_Warp.nii.gz
            fly-001_moving-atlas_fixed-func_scan-001.nii
            fly-001_moving-atlas_fixed-anat.nii
            fly-001_moving-func_fixed-atlas_scan-001.nii
            fly-001_moving-anat_fixed-atlas.nii
            QC/  # directory for QC plots
                fly-001_moving-atlas_fixed-anat_contours.png
        
        report/ # directory for QA report
            report.html
            images/

        logs/ # home for logs generated by dataset-level processes, such as atlasreg
