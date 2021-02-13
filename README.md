# Pytorch tutorials

> Find Notes for the papers [here](https://github.com/SubhadityaMukherjee/DLPapers)

- Many codes are using Pytorch Lightning. More will follow. Eventually I might shift any important ones to it. Why? Well. It did what I wanted to with this repository in terms of standardization. Why be redundant. 
- Make it easy for anyone to understand huge libraries by taking small examples and breaking it up in that way
- End up with boilerplate code for a variety of tasks
- Note that this is work in progress. And many more folders will be added in the coming days

## How to use this repo?

- Look for whats *new*. Since most of the code remains the same, there are only a few bits that change. Find them. And youll understand.
- Search for the "#This is new" comment
- How do you find whats new? Look at the basicNet repo. This is the starter template. Everything comes from there.
- What about GANs? Look at DCGAN. It is the basic one
- Papers? Well you can refer to my notes repo. (above) or you can just read the paper (I recommend this )

## There are a billion similar repositories, what makes this different?

- Want to understand research papers? Check out my other repo [Link](https://github.com/SubhadityaMukherjee/PaperImplementations/)
- Comments!!
- Standardized (mostly) code. Aka I try to be as uniform in my approach as possible
- Might write a few blog posts
- To be very honest, it is for my practise xD

## FAQ

- What is readmegen.sh?? Well its a bash script which takes a template and adds a folder wise listing so its easy to see whats been done. (I am lazy come on.)
- Can I use my own dataset? Yes of course. Just modify the dataloader. (I will be adding tutorials for that soon either way)
- No distributed training. I have only one GPU (sad) so I wont be using it
- What is the accuracy? Well run it and find out. I cant train everything to its maximum capacity due to lack of time
- Why so many files? I wonder the same sometimes. But its easier to modify for other data. And keeps my errors easily identifiable
- Why do you have so many arguments? Most of them arent even used? This is because I wanted to "standardize the codes". So I dont remove much. Just add what I need. That way its easier to see the difference between things. Just remove it if you dont want it there. (If it doesnt break anything)


## References
All the sites I referred to for the codes here (heavily modified so you probably wont find the same things but here for future reference just in case anyone wants it). 
I have to admit that I relied pretty heavily on these repos. Do have a look at them if you are interested.

- [pytorch/examples](https://github.com/pytorch/examples)
- [pytorch/tutorials](https://github.com/pytorch/tutorials/)
- [usuyama](https://github.com/usuyama/pytorch-unet) 
- [GANS](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_div/wgan_div.py)

## What can you find in this repo?


- GAN
- GAN/BGAN
- GAN/CGAN
- GAN/CoGAN(WIP)
- GAN/DCGAN
- GAN/WGANdiv
- GAN/WGANgp
- GAN/draGAN
- GAN/lsGAN
- GAN/relativisticGAN
- TrainingParams
- TrainingParams/BatchnormOnlyBatchnorm
- TrainingParams/FocalLoss
- TrainingParams/Mish
- TrainingParams/Pruning
- TrainingParams/SeLU
- TrainingParams/Swish
- applications
- applications/ASLRecog
- applications/AdversarialFGSM
- applications/AudioClassification
- applications/FederatedLearningPySyft
- applications/LSTMcosineWave
- applications/SuperRes
- applications/basicTextClassification
- applications/leafDiseaseClassification
- applications/simpleRecommendationSystem
- architectures
- architectures/AlexNet
- architectures/MobileNet
- architectures/STN
- architectures/ShuffleNet
- architectures/SqueezeNet
- architectures/Unet
- architectures/VAE
- architectures/VGG16
- architectures/basicNet
- architectures/standardModels
- node_modules
- node_modules/@jupyter-widgets
- node_modules/@jupyterlab
- node_modules/@lumino
- node_modules/@types
- node_modules/ajv
- node_modules/backbone
- node_modules/base64-js
- node_modules/fast-deep-equal
- node_modules/fast-json-stable-stringify
- node_modules/jquery
- node_modules/json-schema-traverse
- node_modules/json5
- node_modules/lodash
- node_modules/minimist
- node_modules/moment
- node_modules/node-fetch
- node_modules/path-browserify
- node_modules/punycode
- node_modules/querystringify
- node_modules/requires-port
- node_modules/underscore
- node_modules/uri-js
- node_modules/url-parse
- node_modules/ws
