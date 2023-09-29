# tarzan_lite

Cli for Number recognition with **Machine Learning**  and **Deep Learning**
**tarzan_model.h5** is my own model with **Convolution Neuronal Network** (CNN)

## How to install

```bash
git clone "https://github.com/j03-dev/tarzan_lite"
cd tarzan_lite 
python -m pip install -r requirement.txt
```

## How to use it

the option `dl` mean you use `deep learning` for the recognition <br>
`ml` option is also exist mean use `machine learning`

#### Deep learning using `CNN`

```bash
python main.py --image image_test.jpg --option dl
```

#### Machine learning algo using `XOR and AND`

```bash
python main.py --image image_test.jpg --option ml
```
