#!/bin/bash

# 実験するパラメータのリスト
deoder_names=("simple", "pixel_shuffle_attention" "context_pyramid")
loss_names=("mse" "bce" "focal" "kldiv")

# 総当たりで実験を実行
for decoder in "${deoder_names[@]}"
do
    for loss in "${loss_names[@]}"
    do
        echo "======================================================================"
        echo "Running experiment with decoder: $decoder and loss: $loss"
        echo "======================================================================"

        # Hydraを使ってパラメータをオーバーライドして学習スクリプトを実行
        python -m development.court_pose.vit_heatmap_01.train \
            model.decoder_name=$decoder \
            training.loss.name=$loss

        # エラーが発生したらスクリプトを停止
        if [ $? -ne 0 ]; then
            echo "Error occurred in experiment with decoder: $decoder and loss: $loss. Exiting."
            exit 1
        fi
    done
done

echo "All experiments completed successfully."
