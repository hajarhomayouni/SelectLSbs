# SelectLSbs
Using LSTM-Autoencoder for Optimal Selection of Least Significant Bits for Watermarking of Electronic Control Unit Packets in Heavy Vehicles
</br>

How to use:</br>
1. Preprocess data: </br>
python3 preprocess.py data.csv PGN SPN "Byte stream"</br>
Example: python3 firstscript.py KenworthData.csv 65263 0 "B5B6" </br>
output: preprocessed.csv </br>

2. Select LSBs:</br>
pythons SelectLSbs.py preprocessed.csv </br>

3. Evaluate using denoising and calculating the difference </br>
ongoing work...

