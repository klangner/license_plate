# CNN based license plate recognition system

Currently there is only license plate detection. Given image this network will find position of the plate on the image.

The best network consist of the following layers:
  * 3x3 Convolution with 32 features
  * Max pooling 2x2
  * 2x2 Convolution with 64 features
  * Max pooling 2x2
  * 2x2 Convolution with 128 features
  * Max pooling 2x2
  * 500 neurons dense
  * 500 neurons dense
  
Best score: 0.025452  

## Notes

  * It looks that the aim is to get score less 0.003
  * After reaching 0.0307 score the training is slowing. Only  50% epoch improve 10^-4 per 10 epoch
  
