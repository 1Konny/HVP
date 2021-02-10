## How to prepare data

### Pre-requisite

1. We assume that you've cloned this repo under `$REPO_DIR`

   ```bash
   REPO_DIR='/path/to/clone/this/repo'
   git clone https://www.github.com/1Konny/hierarchicalvideoprediction $REPO_DIR
   
   ----
   
   ./$REPO_DIR
   |-- assets
   |-- docs
   |-- image_generator
   |-- scripts
   `-- structure_generator
   ```

2. Clone the repo for the semantic segmentation model used in our paper.

   ```bash
   SEMSEG_REPO_DIR='/path/to/clone/this/repo'
   git clone https://github.com/1Konny/semantic-segmentation.git --single-branch --branch sdcnet $SEMSEG_REPO_DIR
   ```

3. Download pretrained weights for the sementic segmentation model ([Cityscapes](https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view?usp=sharing), 
   [Kitti](https://drive.google.com/file/d/1OrTcqH_I3PHFiMlTTZJgBy8l_pladwtg/view?usp=sharing)) to `$SEMSEG_REPO_DIR/pretrained_weights` directory as follows.

   ```bash
   $SEMSEG_REPO_DIR/pretrained_models/cityscapes_best.pth
   $SEMSEG_REPO_DIR/pretrained_models/kitti_best.pth
   ```

   

### KITTI Dataset

1. Go to the [official website](http://www.cvlibs.net/datasets/kitti/raw_data.php), download zip files containing videos, and extract all of them under `$REPO_DIR/datasets_raw/KITTI/images` directory as follows.

   ```
   ./$REPO_DIR/datasets_raw/KITTI/images
   |-- 2011_09_26
   |   |-- 2011_09_26_drive_0001_sync
   |   |-- ...
   |   `-- 2011_09_26_drive_0119_sync
   |-- 2011_09_28
   |   |-- 2011_09_28_drive_0001_sync
   |   |-- ...
   |   `-- 2011_09_28_drive_0225_sync
   |-- 2011_09_29
   |   |-- 2011_09_29_drive_0004_sync
   |   |-- ...
   |   `-- 2011_09_29_drive_0108_sync
   |-- 2011_09_30
   |   |-- 2011_09_30_drive_0016_sync
   |   |-- ...
   |   `-- 2011_09_30_drive_0072_sync
   `-- 2011_10_03
       |-- 2011_10_03_drive_0027_sync
       |-- ...
       `-- 2011_10_03_drive_0058_sync
   ```

2. Extract semantic label maps

   ```bash
   cd $SEMSEG_REPO_DIR
   bash extract_labels.sh KITTI $REPO_DIR/datasets_raw/KITTI/images $REPO_DIR/datasets_raw/KITTI/semantic_labels
   ```

3. Pre-process images and labels

   ```
   cd $REPO_DIR
   python datasets_raw/process_kitti.py
   ```

4. Then, the following directories will be saved:

   ```
   $REPO_DIR/structure_generator/datasets/KITTI_64
   $REPO_DIR/image_generator/datasets/KITTI_vid2vid_90
   ```

   

### Cityscapes Dataset

1. Go to the [official website](https://www.cityscapes-dataset.com/downloads/), download `leftImg8bit_sequence_trainvaltest.zip`, and extract all of them under `$REPO_DIR/datasets_raw/Cityscapes/images` directory as follows.

   ```bash
   ./$REPO_DIR/datasets_raw/Cityscapes/images
   |-- test
   |   |-- berlin
   |   |-- bielefeld
   |   |-- bonn
   |   |-- leverkusen
   |   |-- mainz
   |   `-- munich
   |-- train
   |   |-- aachen
   |   |-- bochum
   |   |-- bremen
   |   |-- cologne
   |   |-- darmstadt
   |   |-- dusseldorf
   |   |-- erfurt
   |   |-- hamburg
   |   |-- hanover
   |   |-- jena
   |   |-- krefeld
   |   |-- monchengladbach
   |   |-- strasbourg
   |   |-- stuttgart
   |   |-- tubingen
   |   |-- ulm
   |   |-- weimar
   |   `-- zurich
   `-- val
       |-- frankfurt
       |-- lindau
       `-- munster
   ```

2. Extract semantic label maps

   ```bash
   cd $SEMSEG_REPO_DIR
   bash extract_labels.sh Cityscapes $REPO_DIR/datasets_raw/Cityscapes/images $REPO_DIR/datasets_raw/Cityscapes/semantic_labels
   ```

3. Pre-process images and labels

   ```bash
   cd $REPO_DIR
   python datasets_raw/process_cityscapes.py
   ```

4. Then, the following directories will be saved:

   ```
   $REPO_DIR/structure_generator/datasets/Cityscapes_256x512
   $REPO_DIR/image_generator/datasets/Cityscapes_256x512
   ```

   
