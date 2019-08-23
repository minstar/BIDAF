make_data:
			python preprocess.py \
			--is_load False \
			--batch_size 15 \

make_train:
			python main.py \
			--is_load True \
			--batch_size 60 \
			--epochs 12 \

for_check:
			nohup python main.py \
			--is_load True \
			--epochs 12 \
			--batch_size 15 \
			> train.out & \
