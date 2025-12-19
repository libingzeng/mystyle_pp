python project_anchor.py \
    --images_dir celebrities1/barack_obama/images_free/save01 \
    --output_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999 \
    --anchor_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/debug/during/anchors_all_dict_3999/w \
    --generator_path celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/debug/during/mystyle_model_step3999.pt \
    --is_wplus False \
    --device 0 \

python train_pti.py \
    --images_dir celebrities1/barack_obama/images_free/save01 \
    --output_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999 \
    --anchor_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999/w \
    --generator_path celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/debug/during/mystyle_model_step3999.pt \
    --tune_steps 100 \
    --device 0 \

python generate_test_pti_yawpitchexpage.py \
    --images_dir celebrities1/barack_obama/images_free/save01 \
    --anchor_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999/w \
    --pti_images_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999/pti_images \
    --output_path celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/application_project_is_wplus0_images_free_3999/ \
    --anchors_path celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/debug/during/anchors_all_dict_3999/w \
    --axis_path celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3/debug/during/anchors_axis_dict.pt \
    --device 0 \
    --align 1 \
    --has_age 0 \
    --alg_type ours \

