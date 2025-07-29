PROJECT_DIR=$PWD
TRACKER_NAME="sstrack"
CONFIG_NAME="baseline_256_60ep"

cd pysot_toolkit

python test.py \
	--dataset VOT2018 \
	--dataset_path /dataset/VOT2018 \
	--name $TRACKER_NAME \
	--cfg_file "$PROJECT_DIR/experiments/$TRACKER_NAME/$CONFIG_NAME.yaml" \
    --snapshot "$PROJECT_DIR/output/checkpoints/train/$TRACKER_NAME/$CONFIG_NAME/SSTrack_ep0150.pth.tar"

python test.py \
	--dataset OTB100 \
	--dataset_path /dataset/otb \
	--name $TRACKER_NAME \
	--cfg_file "$PROJECT_DIR/experiments/$TRACKER_NAME/$CONFIG_NAME.yaml" \
    --snapshot "$PROJECT_DIR/output/checkpoints/train/$TRACKER_NAME/$CONFIG_NAME/SSTrack_ep0150.pth.tar"


python eval.py \
	--dataset VOT2018 \
	--tracker_prefix $TRACKER_NAME

python eval.py \
	--dataset OTB100 \
	--tracker_prefix $TRACKER_NAME
