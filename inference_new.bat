PATH=C:\CompVision\python;C:\CompVision\python\scripts;
@set THEANO_FLAGS=floatX=float32,device=cpu
@set KERAS_BACKEND=tensorflow
python inference_new.py -s=True  -f=models
@pause