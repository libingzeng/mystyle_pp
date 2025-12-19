CUDA_VISIBLE_DEVICES=2 python 0_align_face.py \
    --images_dir celebrities1/barack_obama/images_free/images00 \
    --save_dir celebrities1/barack_obama/images_free/save01 \
    --trash_dir celebrities1/barack_obama/images_free/trash \
    --landmarks_model /data2/libingzeng/view_synthesis/mystyle_plus_plus/third_party/checkpoints/shape_predictor_68_face_landmarks.dat \
    --min_size 100 \
    --min_id_size 50 \
    # --id_model /data2/libingzeng/view_synthesis/mystyle_plus_plus/third_party/checkpoints/model_ir_se50.pth \