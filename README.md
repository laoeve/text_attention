C++ Transformer code without GPU library
==================================================
It is an implementation of Transformer targtes on NLP
The code is originally based on [dianhsu swin-transformer](https://github.com/dianhsu/swin-transformer-cpp.git)

Archived files (i.e., parameters, dictionary, input sentence) can be downaload at [Project files](https://drive.google.com/drive/folders/17wloS01iR_vWAiZKwJCC8_3HH8-uIaxR?usp=sharing)

Build and run 
-------------------------
1. Get the code

        $ git clone <url>
        $ cd <PROJECT DIRECTORY>

2. Download the above archived files and unzip in directory

        $ tar -xzvf attention_files.tar.gz

3. Build the code

        $ mkdir build
        $ cd build
        $ cmake ..
        $ make 
        $ <model binary> <YOUR OPTIONS>

4. To run in release mode, please set CMAKE_BUILD_TYPE as "Release"

Contributors
-----------------------
+ Hyunjun Park     laoeve@capp.snu.ac.kr
+ Hyokeun Lee      hklee@capp.snu.ac.kr

