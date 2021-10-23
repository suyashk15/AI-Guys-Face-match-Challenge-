# Face-match-Challenge

![i1](https://user-images.githubusercontent.com/72187755/138566566-6b20bf9e-636a-4017-ba94-c99472e409b7.jpg) ![i2](https://user-images.githubusercontent.com/72187755/138566622-3bac0928-6a19-45de-bcfa-c2e07f75256e.png) ![i3](https://user-images.githubusercontent.com/72187755/138566671-277d9bc3-9856-4ba8-b576-9cbc97f76ebc.png)

Summary of the workflow:

Using OpenCV, we read the train.csv file entries as images and converted them to grey images, both image1 and image2 entries

Then we used Haarcascade frontalface defalut algorithm of opencv on these images to detect faces and cropped them

We resized the image matrices to (50x50) for unifromity and less processing

The next step was to flatten these matrices to (2500x1)

Then to normalize the values, we divided each value by 255

Then we subtracted the image matrices for each row, (image1 - image2) and passed this as a feature to the neural network where
y values were the labels given in train.csv file

After validating the model on our training data, we got an accuracy of 62%

When the model was used on test.csv file, we got 193 entries matched (label = 1)

