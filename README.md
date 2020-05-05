# SelectLSbs
Using LSTM-Autoencoder for Optimal Selection of Least Significant Bits from Electronic Control Unit Packets in Heavy Vehicles
</br>

**How to use**:</br>
1. Preprocess data: </br>
python3 preprocess.py [data.csv] [PGN] [SPN] ["Byte stream"]</br>
Example: python3 firstscript.py KenworthData.csv 65263 0 "B5B6" </br>
output: preprocessed.csv </br>

2. Select LSBs:</br>
python3 SelectLSbs.py [preprocessed.csv] </br>
output: a printed list of LSbs

3. Evaluate using denoising and calculating the difference </br>
  a) Update the follwoing lines in the evaluate.py code depending on the nature (i.e., PGN and SPN) of the data you want to evaluate: </br>
  #Hard Code Value</br>
  resolution =0.00390625</br>
  offset=0</br>
  unit="MPa"</br> 
  *Note*: This information can be found in the SAE-J1939 standardized protocol for heavy vehicles.</br> 
  b) python3 evaluate.py [preprocessed.csv] [LSbs] [denoising value (i.e., 0 or 1)] </br>
  Example: python3 evaluate.py TU36_002_1_61444_0_B3B4.csv "1,2,3,4,5,6,7" 0 </br>
  Output: A plot showing the actual (blue line) and denoised (orange line) values of data records over time based on the selected LSbs  


