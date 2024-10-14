This github repo contains the final weights of the Deepfake Detection model and a testDeepfake.ipynb file.
For testing the model on your custom image, you just need to download the ipynb file and the weights and change the second last and the last cell with the path to weights and path to test image respectively
(val accuracy ~93%; test accuracy is varying but on an average it comes out to be ~68% on images which is not as per expectations, 
probably the accuracy of the testing on videos might increase upon using the Image Aggregation technique for predicting nature of the video but this is yet to be implemented)
