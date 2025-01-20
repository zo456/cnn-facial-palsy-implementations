# Stroke/Facial Palsy assessment - Implementation of the SOTA

Implementations of the methods proposed by Guo et al. [1]

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
