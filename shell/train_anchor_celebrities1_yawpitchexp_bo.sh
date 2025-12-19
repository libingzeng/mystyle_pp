python train_anchor_yawpitchexp_celebrities1.py \
    --images_dir celebrities1/barack_obama/images_train_renamed \
    --output_dir celebrities1/barack_obama/output_anchor_train/pca_rank_3_wspace_auto_yawpitchexp3 \
    --encoder_checkpoint /data2/libingzeng/view_synthesis/mystyle_plus_plus/third_party/checkpoints/faces_w_encoder.pt \
    --device 0 \
    --pca_rank 3 \
    --is_wspace True \
    --generator_path /data2/libingzeng/view_synthesis/mystyle_plus_plus/third_party/checkpoints/ffhq.pkl \
