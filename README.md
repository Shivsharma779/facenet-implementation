## FaceNet implementation
### [Click here to view the presentation](https://docs.google.com/presentation/d/e/2PACX-1vTihbOz33Oyu4n9txbVQfVdXswTRGtKJV3TwjcKYQpHszRszCh3j8XFooEc0wFaiO6WGFzAoh2WACxU/pub?start=false&loop=false&delayms=5000)
### Problem satement
In this Project we implement FaceNet system, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity.

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
