#!/usr/bin/env bash


# expected environment variables...
export BASEDIR="/opt"
export SOCCERCODE="$BASEDIR/soccerontable"
export DATADIR="$BASEDIR/datadir"
export DETECTRON="$BASEDIR/detectron"
export OPENPOSEDIR="$BASEDIR/openpose"
export MODELPATH="$BASEDIR/model.pth"

export PYTHONPATH="$PYTHONPATH:$SOCCERCODE:$OPENPOSEDIR:$DETECTRON"
##--     Interpreter:                 /usr/bin/python3 (ver 3.5.2)
##--     packages path:               lib/python3.5/dist-packages
 
 
set -e 




## install soccercode
cd $BASEDIR
git clone https://github.com/ngerstle/soccerontable.git $SOCCERCODE

cd ${SOCCERCODE}
chmod u+x compile.sh
./compile.sh

cd ${SOCCERCODE}/utils/nms/
python3 setup.py build_ext --inplace
python2 setup.py build_ext --inplace

cd ${SOCCERCODE}
pip3 install -r requirements.txt
pip2.7 install -r requirements.txt

## get example dataset
mkdir -p $DATADIR
cd $BASEDIR
wget http://grail.cs.washington.edu/projects/soccer/barcelona.zip && unzip barcelona.zip && mv barcelona/*  $DATADIR




############## MANUAL CHANGES #########
# comment out detectron/modeling/detector.py
# -46;+46 : #assert



pip2.7 install pycocotools h5py
pip3 install pycocotools h5py

## run detectron
mkdir -p  $DATADIR/detectron
cd $SOCCERCODE
cp utils/thirdpartyscripts/infer_subimages.py $DETECTRON/tools/
cd $DETECTRON 
python2 tools/infer_subimages.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-dir $DATADIR/detectron --image-ext jpg --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl $DATADIR/images/
echo "DONE: DETECTRON"

# Now we can run the calibration step. In the first frame we give 4 manual correspondences and afterwards the camera parameters are optimized to fit a synthetic 3D field to the lines in the image.

cd $SOCCERCODE
python3 demo/calibrate_video.py --path_to_data $DATADIR
echo "DONE: CALIBRATION"

# Next, we estimate poses, near the bounding boxes that Mask-RCNN gave.
#
#
#TODO add following to soccer3d/core.py in def estimate_poses()
#            def opencv_matrix(loader, node):
#                mapping = loader.construct_mapping(node)
#                mat = np.array(mapping["data"])
#                mat.resize(mapping["rows"], mapping["cols"])
#                return mat
#
#            yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
#
#TODO add if check in soccer3d/core.py def estimate_poses():
#	    posefilepath = join(join(self.path_to_dataset, 'tmp'), '{0}_pose.yml'.format(j))
#                   if os.path.isfile(posefilepath):
#                        with open(posefilepath) as data_file:
#



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
cd $SOCCERCODE
python3 demo/estimate_poses.py --path_to_data $DATADIR --openpose_dir $OPENPOSEDIR
echo "DONE: ESTIMATE POSES"

# The estimated poses cover very well the players in terms of localization/extend/etc. We use them to make individual crops of players for every frame for further processing. We use also the poses to refine the instance segmentation.

cd ${SOCCERCODE}
python3 demo/crop_players.py --path_to_data $DATADIR
echo "DONE: CROP PLAYERS"
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

# Next, we convert the estimated depthmaps to pointclouds.

cd ${SOCCERCODE}
python3 demo/depth_estimation_to_pointcloud.py --path_to_data $DATADIR
echo "DONE: POINTCLOUD"

# Finally we generate one mesh per frame, with the smooth position of the players, based on tracking. Note that the resolution of the mesh is reduced, so later it can easily fit into Hololens.

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
