## Train
* Weights：
<p>&emsp;&emsp;the weights of MSNet256 and MSNet512 in the ITS datasets are released.</p>
<p>&emsp;&emsp;MSNet256：https://drive.google.com/open?id=1H9CqWKZZwLn8V-nOWjCopyBRzKhZDTvo</p>
<p>&emsp;&emsp;MSNet512：https://drive.google.com/open?id=1oRDLw0y8u5JH0B3eN-nQPqzu0IJlU9XX</p>

* MSNet256: 
```bash
python train.py --cuda --gpus=4 --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000 --n 1
```
* MSNet512: 
```bash
python train.py --cuda --gpus=4 --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000 --n 2
```

## Test
```bash
python test.py --cuda --checkpoints=/path/to/checkpoint --test=/path/to/testimages
```
