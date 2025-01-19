This is the code of our paper "Two-stage acoustic modeling for underwater target recognition using raw-waveform."

Two experimental datasets were used to evaluate the UATR method based on the Two-stage acoustic modeling learning proposed in this paper.

We evaluate the proposed model’s performance on the ShipsEar[1] and DeepShip[2] datasets. The noise is superimposed on the raw signal to generate mixed signals with two noise levels (-15 dB, -10 dB). The signal is downsampled to 16 kHz and cropped to 1 second in length. The noise comes from ShipsEar, which provides high-quality real underwater noise and ensures the experiment’s reliability and authenticity.

[1] David Santos-Dom´ınguez, Soledad Torres-Guijarro, Antonio CardenalLopez, and Antonio Pena-Gimenez, “Shipsear: An underwater vessel ´
noise database,” Applied Acoustics, vol. 113, pp. 64–69, 2016.

[2] Muhammad Irfan, ZHENG Jiangbin, Shahid Ali, Muhammad Iqbal, Zafar Masood, and Umar Hamid, “Deepship: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification,” Expert Systems with Applications, vol. 183, pp. 115270, 2021.
