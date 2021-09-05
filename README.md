## FaceNet implementation
### Problem satement
In this Project we implemented FaceNet system, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity.

### [Presentation](https://docs.google.com/presentation/d/e/2PACX-1vTihbOz33Oyu4n9txbVQfVdXswTRGtKJV3TwjcKYQpHszRszCh3j8XFooEc0wFaiO6WGFzAoh2WACxU/pub?start=false&loop=false&delayms=5000)

### Datasets
- [Training Dataset](https://www.kaggle.com/baohoa/modified-vggface2?select=train_refined_resized)
- [Testing Dataset](http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor)

### Steps

- Install all datasets and models using

```sh
bash scripts.sh
```


- Start the training script

```
python train.py
```


Model will be generated under the name `checkpoint.pth`
