#!/usr/bin/env bash


#export PYTHONPATH="$PYTHONPATH:$SOCCERCODE:$OPENPOSEDIR:$DETECTRON"
###--     Interpreter:                 /usr/bin/python3 (ver 3.5.2)
###--     packages path:               lib/python3.5/dist-packages
# 
 
set -e 


## run detectron
cd $DETECTRON 
PYTHONPATH=:$DETECTRON/detectron/:/usr/local/caffe2_build/  python2 tools/infer_subimages.py \
	--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
	--output-dir $DATADIR/detectron \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	$DATADIR/images/

echo "DONE: DETECTRON"


# run the calibration step
cd $SOCCERCODE
python3 demo/calibrate_video.py --path_to_data $DATADIR
echo "DONE: CALIBRATION"


# estimate poses
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
cd $SOCCERCODE
python3 demo/estimate_poses.py --path_to_data $DATADIR --openpose_dir $OPENPOSEDIR
echo "DONE: ESTIMATE POSES"


# make individual crops of players for every frame 
cd ${SOCCERCODE}
python3 demo/crop_players.py --path_to_data $DATADIR
echo "DONE: CROP PLAYERS"

# use the poses to refine instance segmentation.
export OMP_NUM_THREADS=8
./soccer3d/instancesegm/instancesegm --path_to_data $DATADIR/players/ --thresh 1.5 --path_to_model ./soccer3d/instancesegm/model.yml.gz
echo "DONE: instancesegm 2"


# We combine the masks from Mask-RCNN and our pose-based optimization and we prepare the data for the network.
# The model weights can be found here (https://drive.google.com/file/d/1QBLyoNBrFu0oYr15WECzCfOgzuAAQW7w/view?usp=sharing)
cd ${SOCCERCODE}
python3 demo/combine_masks_for_network.py --path_to_data $DATADIR
echo "DONE: COMBINE MASKS"
cd ${SOCCERCODE}
python3 soccer3d/soccerdepth/test.py --path_to_data $DATADIR/players --modelpath $MODELPATH
echo "DONE: test depth"


#  convert the estimated depthmaps to pointclouds.
cd ${SOCCERCODE}
python3 demo/depth_estimation_to_pointcloud.py --path_to_data $DATADIR
echo "DONE: POINTCLOUD"


# generate one mesh per frame, with the smooth position of the players, based on tracking
cd ${SOCCERCODE}
python3 demo/track_players.py --path_to_data $DATADIR
echo "DONE: TRACK PLAYERS"
cd ${SOCCERCODE}
python3 demo/generate_mesh.py --path_to_data $DATADIR
echo "DONE: GEN MESH"


# Just to be sure that everything is fine, we can have a simple opengl visualization
cd ${SOCCERCODE}
python3 demo/simple_visualization.py --path_to_data $DATADIR
echo "DONE:DONE!!!" 
