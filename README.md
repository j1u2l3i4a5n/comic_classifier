# comic classifer 
## usage
     The model used to classify if the comic image is american comic like marvel or japanest comic.
     
## prediction 
- create a folder for saving your input image
- run command `python <image_path> <model_path> <lable_path>`
    - <image_path> is the path of the image
    - <model_path> is a .h5 file model, example model is in the model folder
    - <lable_path> is a text file lable the model's output, example lable is in the model folder
    - example: `python test.jpg model/japan_america.h5 model/japan_america.lable`
- output will print a text that which kind of comic it is

## train model your self
- create a folder for training data
- create a file_path.txt file for recording comics' path, lable, split or not. There is a example file
    - path: a path of a folder save a comic, you can put different chapter in the subfolder of the folder
    - lable: in the project, lable the comic is japanest comic or american comics. You can use other lable if you want
    - split or not: some comic data put two pages in one image file If true, it will cut the image half when input the image.
- (optional) revise the model parameters or model structure in classifier.py





- according to copyright reason, there is no data source in the repository
