# EASE embedding

Run this part of the code to get the EASE embedding. First get the x-vectors from Speechbrain using the code **get_speaker_embedding.py** file. The file will store the x-vectors as specified in line 9 of the code. Also remember to give the path of the wav files in line 8 of this code.

After the x-vectors are stored run **speaker_classifier.py** which will train the EASE model and store the EASE vectos in the folder **EASE_embeddings**. Please set up the paths in lines 120, 122, 124 and 126 of this code accordingly.