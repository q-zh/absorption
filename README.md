
# Single Image Reflection Removal with Absorption Effect
The paper and supplementatry material can be found from [here](http://ci.idm.pku.edu.cn/CVPR21d.pdf).

## Dependencies

- Python 3.5+
- PyTorch 0.4.0+

## Training 

```shell
# Prepare the training 
Please use the provided matlab code (https://github.com/q-zh/absorption/tree/main/matlab).
# Download the VGG model
VGG-16 from http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7
# Run for training
python main.py 
```

## Testing

```shell
# Prepare the test set 
The data folder should include the reflection-contaminated images
# Run for test
python test.py 
```


## Citation
If you find our code is useful, please cite our paper. If you have any problem of implementation or running the code, please contact us: csqianzheng@gmail.com.
```
@inproceedings{CVPR2021_zheng_single,
  title={Single Image Reflection Removal with Absorption Effect},
  author={Zheng, Qian and Shi, Boxin and Chen, Jinnan and Jiang, Xudong and Duan, Ling-Yu and Kot, Alex C},
  booktitle={Proceedings of  Computer Vision and Pattern Recognition},
  year={2021}
}
```
