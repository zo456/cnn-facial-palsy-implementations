# Stroke/Facial Palsy assessment - Implementation of the SOTA

Implementations of the methods proposed by Guo et al. [1], Sajid et al. [2], and Yu et al. [3]. For the method by Yu et al., we recommend using their own official repo at: 
https://github.com/0CTA0/MICCAI20_MMDL_PUBLIC.git

### Environment

#### Python version: 3.8.10

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset directory

The dataset is prepared so that there are directories for each class, with the image frames named following "{patient_id}_{frame_id}" structure.

```
...
|
|---nonstroke
|       |--subject0_frame0.
|       |--subject0_frame1
|       | ...
|       |--subject1_frame0
|       |...
|
|---stroke
        |--patient0_frame0
        |--patient0_frame1
        | ...
        |--patient1_frame0
        |...
```

Run ```img2array_crop.py``` to crop images to face regions and save as arrays.

## Training

Run ```main.py``` to train the model as follows:
```bash 
python main.py --method {['guo', 'sajid', 'yu']} --epoch {NUMBER_EPOCH} --lr {LEARNING_RATE} --save_model {SAVED_WEIGHT_NAME}
```
#### Evaluation

```bash
python validate.py --method {['guo', 'sajid', 'yu']} --load_model {SAVED_WEIGHT_NAME}```
```

#### References
[1] Zhexiao Guo, Minmin Shen, Le Duan, Yongjin Zhou, Jianghuai Xiang, Huijun Ding, Shifeng Chen, Oliver Deussen, and Guo Dan. Deep assessment process: Objective assessment process for unilateral peripheral facial paralysis via deep convolutional neural network. In 2017 IEEE 14th international symposium on biomedical imaging (ISBI 2017), pages 135–138. IEEE, 2017.

[2] Muhammad Sajid, Tamoor Shafique, Mirza Jabbar Aziz Baig, Imran Riaz, Shahid Amin, and Sohaib Manzoor. Automatic grading of palsy using asymmetrical facial features: a study complemented by new solutions. Symmetry, 10(7):242, 2018.

[3] Mingli Yu, Tongan Cai, Xiaolei Huang, Kelvin Wong, John Volpi, James Z Wang, and Stephen TC Wong. Toward rapid stroke diagnosis with multimodal deep learning. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part III 23, pages 616–626. Springer, 2020.
