## text transforemr study

### File explanation
- main.py : check loss values while studying and training
- "main_2.py" : training & eval & save best model to check with test.py
- test.py : see training result. print some expected answer and real ansers
- modle.py : My reproduced transformer model

### How to run
use command below

python test.py

### Accuracies and loss

#### Reproduced(model.py) Trnasformer model Training loss 
![image](https://user-images.githubusercontent.com/48676255/156733761-56c2cf84-0145-49b3-899e-77a29419cbc1.png)
##### Green : with SGD optimizer
##### Blue : with Adam optimizer

#### Transformer from torch.nn VS Reproduced model
![image](https://user-images.githubusercontent.com/48676255/156734013-6141bdcd-0cb6-4b58-91d3-f430e7b4ccc4.png)
##### Green : nn.Transformers with Adam Optimizer
##### Blue : Reproduced model with Adam Optimizer

### Sample Result
![image](https://user-images.githubusercontent.com/48676255/156733141-6c5f0544-c398-4d46-884c-479cb2b36f04.png)
![image](https://user-images.githubusercontent.com/48676255/156733181-77f0e3b0-a225-438e-b3fa-6b29e9726b70.png)
![image](https://user-images.githubusercontent.com/48676255/156733214-3a961c0c-facd-4060-b2e9-dc58ac46c5f9.png)

