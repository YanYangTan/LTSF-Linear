# TSlib

## Models To Be Added
* FEDformer
* Pyraformer
* ETSformer
* Reformer

## For each Model
* Write corresponding scripts

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example

For example:

To train the **DLinear**, you can use the scipt 

```
scripts/EXP-LongForecasting/DLinear.sh
```